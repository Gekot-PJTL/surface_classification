#!/usr/bin/env python3
"""
Grass Range-Profile Data Logger — xWRL1432

Records labeled range-profile ROI vectors to a CSV file for ML training.

Usage:
  1. Run the script, wait for radar data
  2. Point at GRASS   → press 'g' + Enter to start logging grass frames
  3. Point at NOT-GRASS → press 'n' + Enter to start logging not-grass frames
  4. Press 'p' to pause logging (move the sensor, change surface, etc.)
  5. Press 'q' to quit

The CSV is appended to each run, so you can collect data across sessions.
Each row: label, bin0, bin1, ..., binN
"""

import serial
import time
import struct
import threading
import numpy as np
import os
import csv

# ── Radar Config ──────────────────────────────────────────────────
PORT_CLI       = 'COM12'
BAUD_INITIAL   = 115200
CFG_FILE       = 'surface_range.cfg'

MAGIC_WORD         = b'\x02\x01\x04\x03\x06\x05\x08\x07'
PACKET_HEADER_SIZE = 40
TLV_HEADER_SIZE    = 8
TLV_RANGE_PROFILE  = 302
TLV_RANGE_PROFILE_STD = 2

# ── Radar Parameters (from .cfg) ─────────────────────────────────
NUM_ADC_SAMPLES  = 256
SAMPLE_RATE_MHZ  = 12.5
SLOPE_MHZ_PER_US = 160.0
C                = 3e8

NUM_RANGE_BINS   = NUM_ADC_SAMPLES // 2   # 128
RANGE_PER_BIN    = (C * SAMPLE_RATE_MHZ * 1e6) / \
                   (2 * SLOPE_MHZ_PER_US * 1e12 * NUM_ADC_SAMPLES)

# Analysis zone — must match training and inference
NEAR_START_M   = 0.75
NEAR_END_M     = 3.8
NEAR_BIN_START = max(1, int(NEAR_START_M / RANGE_PER_BIN))
NEAR_BIN_END   = min(NUM_RANGE_BINS, int(NEAR_END_M / RANGE_PER_BIN))
ROI_BINS       = NEAR_BIN_END - NEAR_BIN_START

AVG_FRAMES = 10  # moving-average window before logging

# ── Output ────────────────────────────────────────────────────────
LOG_DIR  = "range_profiles"
CSV_FILE = os.path.join(LOG_DIR, "dataset.csv")

# ── Shared State ─────────────────────────────────────────────────
data_lock      = threading.Lock()
latest_profile = None
exit_event     = threading.Event()

label_lock     = threading.Lock()
current_label  = None  # None = paused, 1 = grass, 0 = not grass


# ── Serial Helpers ───────────────────────────────────────────────
def try_open_at_correct_baud():
    for baud in [115200, 1250000, 921600]:
        try:
            ser = serial.Serial(PORT_CLI, baud, timeout=0.5)
            ser.flushInput()
            ser.write(b'sensorStop 0\n')
            time.sleep(0.3)
            resp = ser.read(max(ser.in_waiting, 64))
            text = resp.decode('ascii', errors='replace')
            ratio = sum(1 for c in text if c.isprintable() or c in '\r\n') / max(len(text), 1)
            if ratio > 0.7 and len(text) > 2:
                print(f"  Device responding at {baud} baud")
                return ser, baud
            ser.close()
        except Exception:
            pass
    print("  Falling back to 115200")
    return serial.Serial(PORT_CLI, BAUD_INITIAL, timeout=1), BAUD_INITIAL


def send_config(ser, cfg_path):
    print(f"\n--- Sending config: {cfg_path} ---")
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('%'):
                continue
            print(f"  >> {line}")
            ser.write((line + '\n').encode())
            time.sleep(0.05)
            time.sleep(0.05)
            if ser.in_waiting:
                resp = ser.read(ser.in_waiting).decode('ascii', errors='replace').strip()
                print(f"  << {resp[:80]}")
            if line.startswith('baudRate'):
                parts = line.split()
                if len(parts) >= 2:
                    new_baud = int(parts[1])
                    print(f"\n  *** Switching to {new_baud} baud ***\n")
                    time.sleep(0.3)
                    ser.close()
                    ser = serial.Serial(PORT_CLI, new_baud, timeout=1)
                    time.sleep(0.1)
    print("--- Config sent ---\n")
    return ser


# ── Data Reader Thread ───────────────────────────────────────────
def read_data_stream(ser):
    global latest_profile
    buffer = bytearray()

    while not exit_event.is_set():
        try:
            avail = ser.in_waiting
            if avail:
                buffer += ser.read(avail)
        except Exception:
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
            except Exception:
                buffer = buffer[idx+8:]
                break
            if total_len > 100000 or total_len < PACKET_HEADER_SIZE:
                buffer = buffer[idx+8:]
                break
            if len(buffer) < idx + total_len:
                break

            offset = PACKET_HEADER_SIZE
            for _ in range(num_tlv):
                if idx + offset + TLV_HEADER_SIZE > len(buffer):
                    break
                tlv_type, tlv_length = struct.unpack('<II', buffer[idx+offset:idx+offset+8])
                tlv_data = buffer[idx+offset+8 : idx+offset+8+tlv_length]
                if tlv_type in (TLV_RANGE_PROFILE, TLV_RANGE_PROFILE_STD):
                    num_vals = tlv_length // 4
                    if num_vals >= NUM_RANGE_BINS:
                        vals = struct.unpack(f'<{num_vals}I', tlv_data[:num_vals*4])
                        profile = np.array(vals[:NUM_RANGE_BINS], dtype=np.float64)
                        with data_lock:
                            latest_profile = profile
                offset += 8 + tlv_length

            buffer = buffer[idx + total_len:]
        time.sleep(0.001)


# ── Input Thread ─────────────────────────────────────────────────
def input_thread():
    global current_label
    while not exit_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        with label_lock:
            if cmd == 'g':
                current_label = 1
                print("  >>> LOGGING: GRASS (label=1)")
            elif cmd == 'n':
                current_label = 0
                print("  >>> LOGGING: NOT GRASS (label=0)")
            elif cmd == 'p':
                current_label = None
                print("  >>> PAUSED")
            elif cmd in ('q', 'quit', 'exit'):
                exit_event.set()
                break
        time.sleep(0.05)


# ── Main ─────────────────────────────────────────────────────────
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Check existing data
    existing = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as f:
            existing = sum(1 for _ in f)

    print("=" * 62)
    print("   xWRL1432 Grass Data Logger")
    print("=" * 62)
    print(f"   ROI bins:    {ROI_BINS}  ({NEAR_START_M}–{NEAR_END_M} m)")
    print(f"   Avg window:  {AVG_FRAMES} frames")
    print(f"   Output:      {CSV_FILE}  ({existing} rows so far)")
    print()
    print("   'g' + Enter  →  label frames as GRASS")
    print("   'n' + Enter  →  label frames as NOT GRASS")
    print("   'p' + Enter  →  pause logging")
    print("   'q' + Enter  →  quit")
    print("=" * 62)

    ser, _ = try_open_at_correct_baud()
    time.sleep(0.2)
    ser = send_config(ser, CFG_FILE)

    threading.Thread(target=read_data_stream, args=(ser,), daemon=True).start()
    threading.Thread(target=input_thread, daemon=True).start()

    print("  Waiting for data...", end='', flush=True)
    for _ in range(100):
        with data_lock:
            if latest_profile is not None:
                break
        time.sleep(0.1)
    print(" OK\n")

    from collections import deque
    avg_buffer = deque(maxlen=AVG_FRAMES)
    logged_count = {0: 0, 1: 0}
    frame = 0

    csv_handle = open(CSV_FILE, 'a', newline='')
    writer = csv.writer(csv_handle)

    try:
        while not exit_event.is_set():
            with data_lock:
                profile = latest_profile.copy() if latest_profile is not None else None
            if profile is None:
                time.sleep(0.05)
                continue

            frame += 1
            profile_db = 10 * np.log10(profile + 1)
            avg_buffer.append(profile_db)
            avg_profile = np.mean(avg_buffer, axis=0)

            with label_lock:
                lbl = current_label

            if lbl is not None and len(avg_buffer) >= AVG_FRAMES:
                roi = avg_profile[NEAR_BIN_START:NEAR_BIN_END]
                writer.writerow([lbl] + [f"{v:.4f}" for v in roi])
                logged_count[lbl] += 1

                if frame % 20 == 0:
                    csv_handle.flush()
                    total_g = logged_count[1]
                    total_n = logged_count[0]
                    tag = "GRASS" if lbl == 1 else "NOT GRASS"
                    print(f"  [{tag}]  grass={total_g}  not_grass={total_n}", end='\r')

            time.sleep(0.03)
    finally:
        csv_handle.close()
        total_g = logged_count[1]
        total_n = logged_count[0]
        print(f"\n\nDone. Logged {total_g} grass + {total_n} not-grass = {total_g+total_n} new rows")
        print(f"Saved to: {CSV_FILE}")
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
