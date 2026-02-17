#!/usr/bin/env python3
"""
Range-Doppler Heatmap Reader for xWRL1432 (mmwave_demo firmware)

On xWRL1432, both CLI commands and TLV data use the SAME UART port
(XDS110 Application/User UART). The baudRate command switches that
port's speed, so we reopen at the new rate before reading data.

xWRL1432 only supports baudRate = 1250000.
"""

import serial
import time
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ----- Port config -----
PORT_CLI = 'COM12'          # XDS110 Application/User UART (CLI + data)
BAUD_INITIAL = 115200       # Default CLI baud rate at power-on
BAUD_DATA = 1250000         # xWRL1432 only supports this value
CFG_FILE = 'range_doppler.cfg'

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
PACKET_HEADER_SIZE = 40
TLV_HEADER_SIZE = 8

# xWRL1432 mmwave_demo TLV types (MMWAVE-L-SDK extended message format)
# Check your SDK version — these are the common ones:
TLV_DETECTED_POINTS = 301
TLV_RANGE_PROFILE   = 302
TLV_STATS           = 306
TLV_RANGE_DOPPLER   = 305   # Range-Doppler heatmap (extended msg)

# Also check for the older/standard TLV type numbering:
TLV_RANGE_DOPPLER_STD = 5


def try_open_at_correct_baud():
    """
    Auto-detect current baud rate. The device might still be at 1250000
    from a previous session that didn't cleanly stop.
    """
    for baud in [115200, 1250000, 921600]:
        try:
            ser = serial.Serial(PORT_CLI, baud, timeout=0.5)
            ser.flushInput()
            ser.write(b'sensorStop 0\n')
            time.sleep(0.3)
            resp = ser.read(max(ser.in_waiting, 64))
            text = resp.decode('ascii', errors='replace')

            printable_ratio = sum(1 for c in text if c.isprintable() or c in '\r\n') / max(len(text), 1)

            if printable_ratio > 0.7 and len(text) > 2:
                print(f"Device responding at {baud} baud")
                print(f"  Response: {text.strip()[:80]}")
                return ser, baud
            else:
                print(f"Tried {baud} — garbled ({printable_ratio:.0%} printable)")
                ser.close()
        except Exception as e:
            print(f"Tried {baud} — error: {e}")

    print("Falling back to 115200")
    return serial.Serial(PORT_CLI, BAUD_INITIAL, timeout=1), BAUD_INITIAL


def send_config(ser, cfg_path):
    """
    Send config line-by-line. When we hit 'baudRate', close and reopen
    the port at the new baud rate.
    """
    print(f"\n--- Sending config: {cfg_path} ---")

    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('%'):
                continue

            print(f"  >> {line}")
            ser.write((line + '\n').encode())
            time.sleep(0.05)

            # Read response
            time.sleep(0.05)
            if ser.in_waiting:
                resp = ser.read(ser.in_waiting)
                text = resp.decode('ascii', errors='replace').strip()
                print(f"  << {text[:80]}")

            # Handle baudRate switch
            if line.startswith('baudRate'):
                parts = line.split()
                if len(parts) >= 2:
                    new_baud = int(parts[1])
                    print(f"\n  *** Switching to {new_baud} baud ***\n")
                    time.sleep(0.3)
                    ser.close()
                    ser = serial.Serial(PORT_CLI, new_baud, timeout=1)
                    time.sleep(0.1)

    print("--- Config sent. Sensor should be running. ---\n")
    return ser


def parse_and_display_heatmap(tlv_data, num_range_bins, num_doppler_bins, ax, fig):
    """Parse range-Doppler heatmap TLV and update the plot."""
    expected_size = num_range_bins * num_doppler_bins

    # Try 32-bit unsigned (MMWAVE-L-SDK uses uint32)
    if len(tlv_data) == expected_size * 4:
        vals = np.array(struct.unpack(f'<{expected_size}I', tlv_data), dtype=np.float64)
    # Try 16-bit unsigned (older SDK format)
    elif len(tlv_data) == expected_size * 2:
        vals = np.array(struct.unpack(f'<{expected_size}H', tlv_data), dtype=np.float64)
    else:
        print(f"  Unexpected heatmap size: {len(tlv_data)} bytes "
              f"(expected {expected_size*4} or {expected_size*2})")
        return

    heatmap = vals.reshape((num_range_bins, num_doppler_bins))

    # Shift zero-Doppler to center
    heatmap = np.fft.fftshift(heatmap, axes=1)

    ax.clear()
    # Use log scale for better visualization; add 1 to avoid log(0)
    im = ax.imshow(heatmap + 1, aspect='auto', origin='lower',
                   norm=LogNorm(), cmap='jet',
                   extent=[-num_doppler_bins//2, num_doppler_bins//2,
                           0, num_range_bins])
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    ax.set_title('Range-Doppler Heatmap')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def listen_for_data(ser, num_range_bins=64, num_doppler_bins=32):
    """Listen for TLV frames and display range-Doppler heatmap."""
    print(f"--- Listening for TLV data at {ser.baudrate} baud ---")
    print(f"    Expecting heatmap: {num_range_bins} range x {num_doppler_bins} Doppler bins")
    print(f"    Magic word: {MAGIC_WORD.hex(' ')}\n")

    # Set up live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('xWRL1432 Range-Doppler Heatmap (waiting for data...)')
    plt.show(block=False)

    buffer = bytearray()
    frame_count = 0

    while True:
        try:
            avail = ser.in_waiting
            if avail:
                buffer += ser.read(avail)
        except Exception as e:
            print(f"Read error: {e}")
            break

        while True:
            idx = buffer.find(MAGIC_WORD)
            if idx == -1:
                if len(buffer) > 16384:
                    buffer = buffer[-4096:]
                break

            if len(buffer) < idx + PACKET_HEADER_SIZE:
                break

            try:
                version, total_len, platform, frame_num, cpu, num_obj, num_tlv, subf = \
                    struct.unpack('<IIIIIIII', buffer[idx+8:idx+40])
            except struct.error:
                buffer = buffer[idx+8:]
                break

            if total_len > 200000 or total_len < PACKET_HEADER_SIZE:
                buffer = buffer[idx+8:]
                break

            if len(buffer) < idx + total_len:
                break

            frame_count += 1

            # Parse TLVs
            offset = PACKET_HEADER_SIZE
            got_heatmap = False

            for t in range(num_tlv):
                if idx + offset + TLV_HEADER_SIZE > len(buffer):
                    break

                tlv_type, tlv_length = struct.unpack(
                    '<II', buffer[idx+offset:idx+offset+8])
                tlv_data = buffer[idx+offset+8 : idx+offset+8+tlv_length]

                if tlv_type in (TLV_RANGE_DOPPLER, TLV_RANGE_DOPPLER_STD):
                    print(f"[Frame {frame_num}] Range-Doppler heatmap: "
                          f"{tlv_length} bytes (TLV type {tlv_type})")
                    parse_and_display_heatmap(
                        tlv_data, num_range_bins, num_doppler_bins, ax, fig)
                    fig.suptitle(f'xWRL1432 Range-Doppler Heatmap — Frame {frame_num}')
                    got_heatmap = True
                else:
                    print(f"[Frame {frame_num}] TLV type={tlv_type}, len={tlv_length}")

                offset += 8 + tlv_length

            if not got_heatmap and frame_count <= 5:
                print(f"[Frame {frame_num}] No heatmap TLV in this frame "
                      f"(got {num_tlv} TLVs, {num_obj} objects)")

            buffer = buffer[idx + total_len:]

        # Keep matplotlib responsive
        plt.pause(0.01)


def main():
    print("=" * 60)
    print("  xWRL1432 Range-Doppler Heatmap Viewer")
    print("=" * 60)

    # Step 1: Detect current baud rate and stop sensor
    ser, _ = try_open_at_correct_baud()
    time.sleep(0.2)

    # Step 2: Send config (handles baudRate switch)
    ser = send_config(ser, CFG_FILE)

    # Step 3: Listen and plot
    # These should match sigProcChainCfg in your .cfg:
    #   sigProcChainCfg 64 32 ...  →  64 range bins, 32 Doppler bins
    # (first arg = range FFT size → numRangeBins = FFT_size / 2 for real ADC)
    # The xWRL1432 SDK doc says numRangeBins = half of range FFT size
    # With 256 ADC samples and 64 in sigProcChainCfg → likely 32 range bins
    # Adjust if needed based on actual data size
    listen_for_data(ser, num_range_bins=128, num_doppler_bins=32)


if __name__ == "__main__":
    main()