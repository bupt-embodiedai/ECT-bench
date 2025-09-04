import serial
import time
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import pandas as pd
from threading import Thread, Lock
from collections import deque
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
# ===== Configuration Parameters =====
STM32_PORT = 'COM14'  # STM32 serial port
CAP_PORT = 'COM11'  # Capacitance board serial port
BAUDRATE = 115200  # Baud rate for both devices
TOTAL_STEPS = 1024  # Total steps
SAMPLES_PER_POSITION = 20  # Samples per position
OUTPUT_DIR = '0805DATA\\resin\\ect_dataC'  # Output data directory
CALIBRATION_SAMPLES = 50  # Background calibration samples
COLLECT_TIMEOUT = 25.0  # Collection timeout
IMAGING_FREQUENCY = 5  # Imaging update frequency

# Capacitance measurement parameters
SENSORS_TO_USE = 36  # Number of sensors to use
zerop_value = 8388608
DAC_value = 0
RANGE = 8

# Imaging parameters
SMOOTHING_SIGMA = 0.7  # Gaussian smoothing parameter

# ===== Imaging Global Variables =====
S_matrix = None
cfull = None
pixel_count = 0
background_avg = None  # Background capacitance average
background_samples = []  # Background calibration samples
latest_image = np.zeros(1)  # Current reconstructed image
cap_data_history = deque(maxlen=50)  # Recent capacitance values
diff_data_history = deque(maxlen=SENSORS_TO_USE)  # Differential values history
imaging_lock = Lock()

# ===== Shared State =====
visualization_data = {
    "current_step": 0,
    "current_cap_data": None,
    "current_image": None,
    "running": True,
    "background_collected": False,
    "background_available": False,
    "acquisition_running": False
}
data_lock = Lock()


# ===== Data Processing Functions =====
def convert(data):
    """Convert raw data to capacitance values"""
    normalized = (data - zerop_value) / zerop_value
    dac_compensation = 0.133 * (DAC_value / 4)
    return (normalized + dac_compensation) * RANGE


def bytes_to_uint32_le(bytes_data):
    return int.from_bytes(bytes_data, byteorder='little', signed=False)


def parse_capacitance_packet(packet):
    """Parse capacitance data packet"""
    try:
        sensor_count = packet[5]
        # Basic length check
        if len(packet) < 6 + sensor_count * 4 + 2:
            return None

        # Process only needed sensors
        converted_values = []
        for i in range(sensor_count):
            start = 6 + i * 4
            raw_data = bytes_to_uint32_le(packet[start:start + 4])
            converted_values.append(convert(raw_data))

        if len(converted_values) >= 39:
            converted_values = converted_values[3:3 + SENSORS_TO_USE]
        # Pad missing sensors
        while len(converted_values) < SENSORS_TO_USE:
            converted_values.append(0.0)

        return converted_values

    except Exception as e:
        print(f"Parsing error: {e}")
        return None


# ===== Imaging Functions =====
def create_colormap():
    """Create custom blue-white-red colormap"""
    colors = [(0, 0, 1), (0.5, 0.5, 1), (1, 1, 1), (1, 0.5, 0.5), (1, 0, 0)]
    return LinearSegmentedColormap.from_list('ect_cmap', colors, N=256)


def load_imaging_data(s_path, cfull_path):
    """Load sensitivity matrix and full capacitance data"""
    global S_matrix, cfull, pixel_count

    # Skip first three columns
    s_data = pd.read_csv(s_path, header=None).values[:, 3:3 + SENSORS_TO_USE]
    S_matrix = s_data.T
    pixel_count = S_matrix.shape[1]

    cfull_data = pd.read_csv(cfull_path, header=None).values[:, 0:0 + SENSORS_TO_USE]
    cfull = cfull_data[0]  # Use first frame

    epsilon = 1e-10
    row_norms = np.linalg.norm(S_matrix, axis=1, keepdims=True)
    S_matrix = S_matrix / (row_norms + epsilon)

    print(f"Imaging data loaded: S_matrix shape={S_matrix.shape}, cfull shape={cfull.shape}")
    print(f"Pixel count: {pixel_count}")
    return S_matrix, cfull


def lbp_realtime_imaging(c_diff):
    """Perform real-time LBP imaging"""
    global background_avg
    epsilon = 1e-10

    # 1. Calculate noise threshold
    noise_level = np.max(np.abs(c_diff)) * 0.05 if np.max(np.abs(c_diff)) > 0 else 0.001

    # 2. Apply noise threshold
    c_diff_filtered = np.where(np.abs(c_diff) < noise_level, 0, c_diff)

    # 3. Calculate normalized differential capacitance
    if background_avg is not None:
        # denominator = cfull - background_avg
        denominator =  background_avg
        denominator[denominator < epsilon] = epsilon
        c_norm = c_diff_filtered / denominator
    else:
        # If no background data, use differential directly
        c_norm = c_diff_filtered

    # 4. LBP imaging
    image = c_norm @ S_matrix

    # 5. Apply Gaussian smoothing
    if SMOOTHING_SIGMA > 0:
        image = gaussian_filter(image, sigma=SMOOTHING_SIGMA)

    # 6. Normalization
    p2 = np.percentile(image, 2) if len(image) > 0 else 0
    p98 = np.percentile(image, 98) if len(image) > 0 else 1
    image_normalized = (image - p2) / (p98 - p2 + epsilon)
    image_normalized = np.clip(image_normalized, 0, 1)

    return image_normalized


def update_imaging(sample):
    """Update imaging data"""
    global background_avg, latest_image

    if background_avg is None:
        return

    current_values = np.array(sample)
    diff_values = current_values - background_avg

    # Add to history
    cap_data_history.append(current_values)
    diff_data_history.append(diff_values)

    # Update imaging periodically
    if len(cap_data_history) % IMAGING_FREQUENCY == 0:
        # Reconstruct image with current sample
        with imaging_lock:
            latest_image = lbp_realtime_imaging(diff_values)


# ===== Visualization Thread =====
def visualization_thread():
    """Setup real-time visualization (no plt.show() here)"""
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(14, 9), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    ax1 = plt.subplot(gs[0])
    line, = ax1.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title("ECT Differential Capacitance (Raw - Background)", fontsize=14)
    ax1.set_xlabel("Sensor Number", fontsize=12)
    ax1.set_ylabel("Capacitance Difference (pF)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(np.arange(0, SENSORS_TO_USE, 6))
    ax1.set_xticklabels([f"S{i+1}" for i in range(0, SENSORS_TO_USE, 6)])
    ax1.set_xlim(0, SENSORS_TO_USE-1)
    ax1.set_ylim(-0.1, 0.1)

    status_text = ax1.text(0.02, 0.95, "Status: Idle",
                           transform=ax1.transAxes, fontsize=12,
                           bbox=dict(facecolor='lightblue', alpha=0.5))

    ax2 = plt.subplot(gs[1])
    cmap = create_colormap()
    grid_size = int(np.sqrt(pixel_count)) if pixel_count > 0 else 10
    init_image = np.zeros((grid_size, grid_size))
    im = ax2.imshow(init_image, cmap=cmap, vmin=0, vmax=1,
                    interpolation='bicubic', origin='lower')
    ax2.set_title("ECT Real-time Imaging", fontsize=14)
    ax2.set_xlabel("X-axis (Pixels)", fontsize=12)
    ax2.set_ylabel("Y-axis (Pixels)", fontsize=12)
    cbar = plt.colorbar(im, ax=ax2, label='Relative Permittivity')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Air', 'Transition', 'Object'])

    plt.suptitle("Electrical Capacitance Tomography System - Data Acquisition & Real-time Imaging",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    def update_frame(frame):
        with data_lock:
            status = status_text.get_text()
            bg_available = visualization_data["background_available"]
            acq_running = visualization_data["acquisition_running"]
            current_cap = visualization_data["current_cap_data"]

        if acq_running:
            step = visualization_data["current_step"]
            status = f"Step: {step+1}/{TOTAL_STEPS} | Acquisition running..."
        else:
            status = "Status: Ready to acquire (Place sample)" if bg_available else "Status: Collecting background data..."
        status_text.set_text(status)

        if current_cap is not None and len(current_cap) == SENSORS_TO_USE and background_avg is not None:
            diff_values = np.array(current_cap) - background_avg
            x_data = np.arange(SENSORS_TO_USE)
            line.set_data(x_data, diff_values)
            cap_min, cap_max = np.min(diff_values), np.max(diff_values)
            margin = max(0.1, (cap_max - cap_min) * 0.1)
            ax1.set_ylim(cap_min - margin, cap_max + margin)

        with imaging_lock:
            if latest_image is not None and len(latest_image) > 1:
                grid_size = int(np.sqrt(len(latest_image)))
                if grid_size * grid_size == len(latest_image):
                    image_2d = latest_image.reshape((grid_size, grid_size))
                    im.set_data(image_2d)
                    ax2.set_xlim(-0.5, grid_size - 0.5)
                    ax2.set_ylim(-0.5, grid_size - 0.5)

        return line, im, status_text

    ani = animation.FuncAnimation(fig, update_frame, interval=300, blit=False, cache_frame_data=False)

    return fig, ani

# ===== System Control Functions =====
def setup_serial_ports():
    """Initialize serial ports"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Handle serial connections
    try:
        stm32_ser = serial.Serial(STM32_PORT, BAUDRATE, timeout=1)
        time.sleep(1.5)
        stm32_ser.reset_input_buffer()
        stm32_ser.reset_output_buffer()
        print(f"Connected to STM32: {STM32_PORT}")
    except Exception as e:
        print(f"STM32 connection error: {e}")
        stm32_ser = None

    try:
        cap_ser = serial.Serial(CAP_PORT, BAUDRATE, timeout=0.1)
        time.sleep(1.5)
        cap_ser.reset_input_buffer()
        cap_ser.reset_output_buffer()
        print(f"Connected to capacitance board: {CAP_PORT}")
    except Exception as e:
        print(f"Capacitance board connection error: {e}")
        if 'stm32_ser' in locals() and stm32_ser.is_open:
            stm32_ser.close()
        cap_ser = None

    return stm32_ser, cap_ser


def collect_capacitance_samples(cap_ser, num_samples, timeout=COLLECT_TIMEOUT, background_mode=False):
    """Collect capacitance samples"""
    if not cap_ser or not cap_ser.is_open:
        return []

    samples = []
    buffer = bytearray()
    start_time = time.time()
    packets = 0
    good_packets = 0

    if background_mode:
        print(f"Collecting background samples ({num_samples}):", end=" ", flush=True)
    else:
        print(f"Collecting samples ({num_samples}):", end=" ", flush=True)

    while len(samples) < num_samples and time.time() - start_time < timeout:
        # Check if should stop acquisition
        with data_lock:
            if not visualization_data["running"]:
                print("Visualization closed, stopping acquisition")
                break

        # Read serial data
        data = cap_ser.read(cap_ser.in_waiting or 10)
        if data:
            buffer.extend(data)

        # Process data packets
        while buffer:
            # Find packet header
            if len(buffer) < 2:
                break

            header_pos = -1
            for i in range(len(buffer) - 1):
                if buffer[i] == 0x55 and buffer[i + 1] == 0xAA:
                    header_pos = i
                    break

            if header_pos == -1:
                # Remove invalid data
                buffer = buffer[-2:] if len(buffer) > 2 else buffer
                break

            # Check packet length
            if len(buffer) - header_pos < 6:
                break

            sensor_count = buffer[header_pos + 5]
            packet_len = 6 + sensor_count * 4 + 2

            if len(buffer) - header_pos < packet_len:
                break

            # Extract packet
            packet = bytes(buffer[header_pos:header_pos + packet_len])
            del buffer[header_pos:header_pos + packet_len]
            packets += 1

            # Verify checksum
            calc_checksum = sum(packet[:-2]) & 0xFFFF
            recv_checksum = packet[-2] | (packet[-1] << 8)

            if calc_checksum != recv_checksum:
                print("C", end="", flush=True)  # C for checksum fail
                continue

            # Parse data
            sample = parse_capacitance_packet(packet)
            if sample:
                samples.append(sample)
                good_packets += 1
                print(".", end="", flush=True)

                # Update visualization (only for non-background)
                if not background_mode:
                    with data_lock:
                        visualization_data["current_cap_data"] = sample
                    update_imaging(sample)

                if len(samples) >= num_samples:
                    break
            else:
                print("P", end="", flush=True)  # P for parse fail

    # Collection summary
    print(f"\nCollection completed: {len(samples)}/{num_samples} samples")
    print(f"Packets received: {packets}, Valid packets: {good_packets}")

    # Pad missing samples
    while len(samples) < num_samples:
        samples.append([0.0] * SENSORS_TO_USE)
        print("Z", end="", flush=True)  # Z for zero padding

    return samples


def send_step_command_and_wait(stm32_ser):
    """Send step command and wait for completion"""
    if not stm32_ser or not stm32_ser.is_open:
        return False
    # stm32_ser.write(b'r')
    stm32_ser.write(b's')
    stm32_ser.flush()
    print("Movement command sent")

    # Wait for completion signal
    start_time = time.time()
    timeout = 15.0

    while time.time() - start_time < timeout:
        with data_lock:
            if not visualization_data["running"]:
                return False

        if stm32_ser.in_waiting > 0:
            response = stm32_ser.read(1)
            if response == b'0':
                return True
            else:
                # Show debug info only
                if response not in (b'\r', b'\n'):
                    print(response.decode('ascii', 'ignore'), end="", flush=True)
        time.sleep(0.01)

    print("Movement timeout")
    return False


def save_background_data(samples):
    """Save background data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"background_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w') as f:
        f.write(f"# Background Data\n")
        f.write(f"# Samples: {len(samples)}\n")
        f.write(f"# Date: {timestamp}\n")
        headers = ["SampleID"] + [f"Cap_{i}" for i in range(SENSORS_TO_USE)]
        f.write(",".join(headers) + "\n")

        for i, sample in enumerate(samples):
            row = [str(i)] + [f"{v:.8f}" for v in sample]
            f.write(",".join(row) + "\n")

    print(f"Background data saved: {filename} ({len(samples)} samples)")
    return filepath


def save_step_data(step, samples):
    """Save step data and imaging results"""
    global background_avg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_{step:03d}_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Save raw capacitance data
    with open(filepath, 'w') as f:
        f.write(f"# Step: {step}\n")
        f.write(f"# Samples: {len(samples)}\n")
        f.write(f"# Date: {timestamp}\n")
        f.write(f"# Background AVG: {background_avg.tolist()}\n")
        headers = ["SampleID"] + [f"Cap_{i}" for i in range(SENSORS_TO_USE)]
        f.write(",".join(headers) + "\n")

        for i, sample in enumerate(samples):
            row = [str(i)] + [f"{v:.8f}" for v in sample]
            f.write(",".join(row) + "\n")

    print(f"Raw capacitance data saved: {filename} ({len(samples)} samples)")

    # Calculate and save differential capacitance
    if background_avg is not None:
        diff_filename = f"diff_step_{step:03d}_{timestamp}.csv"
        diff_filepath = os.path.join(OUTPUT_DIR, diff_filename)

        with open(diff_filepath, 'w') as f:
            f.write(f"# Step: {step}\n")
            f.write(f"# Samples: {len(samples)}\n")
            f.write(f"# Date: {timestamp}\n")
            f.write(f"# Background AVG: {background_avg.tolist()}\n")
            headers = ["SampleID"] + [f"Diff_{i}" for i in range(SENSORS_TO_USE)]
            f.write(",".join(headers) + "\n")

            for i, sample in enumerate(samples):
                diff_sample = np.array(sample) - background_avg
                row = [str(i)] + [f"{v:.8f}" for v in diff_sample]
                f.write(",".join(row) + "\n")

        print(f"Differential capacitance data saved: {diff_filename}")

    # Save image
    with imaging_lock:
        if len(latest_image) > 1:
            grid_size = int(np.sqrt(len(latest_image)))


            if grid_size * grid_size == len(latest_image):
                image_2d = latest_image.reshape((grid_size, grid_size))

                # Resize from 101x101 to 100x100 using bilinear interpolation
                resized_image = zoom(image_2d, (100 / grid_size, 100 / grid_size), order=1)

                image_filename = f"image_step_{step:03d}_{timestamp}.png"
                image_path = os.path.join(OUTPUT_DIR, image_filename)
                plt.imsave(image_path, resized_image, cmap=create_colormap(), vmin=0, vmax=1, origin='lower')

                print(f"Reconstructed image saved: {image_filename} (resized to 100x100)")

# ===== Main Acquisition Function =====
def run_acquisition(stm32_ser, cap_ser):
    """Main acquisition loop"""
    global background_avg, background_samples

    print("\n" + "=" * 50)
    print("ECT Data Acquisition System - with Background Calibration & Real-time Imaging")
    print("=" * 50)
    print(f"Total steps: {TOTAL_STEPS}")
    print(f"Samples per position: {SAMPLES_PER_POSITION}")
    print(f"Collection timeout: {COLLECT_TIMEOUT} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)

    # 1. Collect background data
    print("Collecting background data (ensure no sample is present)...")
    with data_lock:
        visualization_data["background_collected"] = True

    background_samples = collect_capacitance_samples(
        cap_ser, CALIBRATION_SAMPLES, background_mode=True
    )

    if background_samples:
        # Calculate background average
        background_avg = np.mean(background_samples, axis=0)

        # Save background data
        save_background_data(background_samples)

        with data_lock:
            visualization_data["background_available"] = True

        print("Background data collection completed, average calculated and saved")

        # Wait for user to place sample
        print("\n" + "=" * 50)
        print("Place your sample in position...")
        input("Press Enter to start acquisition...")
    else:
        print("Background collection failed, acquisition aborted")
        return

    # 2. Main acquisition loop
    with data_lock:
        visualization_data["acquisition_running"] = True

    for step in range(TOTAL_STEPS):
        with data_lock:
            visualization_data["current_step"] = step
            if not visualization_data["running"]:
                print("User aborted acquisition")
                break

        print(f"\n>>> Step {step + 1}/{TOTAL_STEPS} <<<")

        # Move stepper motor
        move_start = time.time()

        if not send_step_command_and_wait(stm32_ser):
            print("Movement failed! Aborting acquisition")
            break
        move_time = time.time() - move_start
        print(f"Movement completed: {move_time:.2f} seconds")

        # Wait for system stabilization
        time.sleep(0.8)

        # Collect capacitance data
        cap_start = time.time()
        samples = collect_capacitance_samples(
            cap_ser, SAMPLES_PER_POSITION, background_mode=False
        )
        cap_time = time.time() - cap_start

        # Save data
        save_step_data(step, samples)
        print(f"Collection time: {cap_time:.2f} seconds ({len(samples)} samples)")
        print(f"Step duration: {time.time() - move_start:.2f} seconds")

    print("\nAcquisition completed!")
    with data_lock:
        visualization_data["acquisition_running"] = False


def main():
    # 1. Setup serial ports
    stm32_ser, cap_ser = setup_serial_ports()
    if cap_ser is None:
        return

    # 2. Load imaging data
    S_PATH = "EE15.csv"
    CFULL_PATH = "cfull7.5.csv"
    load_imaging_data(S_PATH, CFULL_PATH)

    # 3. Setup visualization
    fig, ani = visualization_thread()  # Build plot in main thread

    # 4. Start acquisition in background thread
    stm32_ser.write(b'r')  # Reset STM32 motor position if needed
    acq_thread = Thread(target=run_acquisition, args=(stm32_ser, cap_ser), daemon=True)
    acq_thread.start()

    # 5. Show GUI in main thread (this avoids set_wakeup_fd errors)
    plt.show()

    # 6. Cleanup after GUI closed
    with data_lock:
        visualization_data["running"] = False

    if stm32_ser and stm32_ser.is_open:
        stm32_ser.close()
    if cap_ser and cap_ser.is_open:
        cap_ser.close()

    print("System shutdown")
    time.sleep(1)



if __name__ == "__main__":
    main()