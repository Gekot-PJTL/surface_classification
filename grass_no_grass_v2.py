#!/usr/bin/env python3
"""
This code adds auto bin shift based on if sensor height changes

Record a grass reference profile, then continuously classify
whether the live profile matches grass or not.

Uses cosine similarity on the dB-domain range profile within the
region of interest — simple shape comparison, no statistical features.

Usage:
  1. Point radar at grass, press 'c' to record the reference
  2. Move to other surfaces — dashboard shows GRASS / NOT GRASS
  3. Press 'r' to re-record the reference
  4. Press 'q' to quit
"""

import serial
import time
import struct
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

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
MAX_RANGE        = RANGE_PER_BIN * NUM_RANGE_BINS
RANGE_AXIS       = np.arange(NUM_RANGE_BINS) * RANGE_PER_BIN

# Analysis zone
NEAR_START_M   = 0.75
NEAR_END_M     = 2
NEAR_BIN_START = max(1, int(NEAR_START_M / RANGE_PER_BIN))
NEAR_BIN_END   = min(NUM_RANGE_BINS, int(NEAR_END_M / RANGE_PER_BIN))

# Classification
GRASS_THRESHOLD     = 0.89
HEIGHT_TOLERANCE_CM = 5.0   # max expected sensor height change — tune this
MAX_SHIFT_BINS      = max(1, int(np.ceil((HEIGHT_TOLERANCE_CM / 100.0) / RANGE_PER_BIN)))
AVG_FRAMES          = 10    # number of frames for moving average

# ── Shared State ─────────────────────────────────────────────────
data_lock      = threading.Lock()
latest_profile = None
exit_event     = threading.Event()
capture_event  = threading.Event()

grass_ref_db   = None   # recorded grass profile (dB, full length)


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
        except:
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
        except:
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
            except:
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


# ── Classification ───────────────────────────────────────────────
def cosine_similarity(a, b):
    """Cosine similarity between two vectors. Returns 0–1."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


def classify_grass(live_db, ref_db):
    """Compare live ROI against grass reference with small shift alignment."""
    live_roi = live_db[NEAR_BIN_START:NEAR_BIN_END]
    ref_roi  = ref_db[NEAR_BIN_START:NEAR_BIN_END]
    n = len(live_roi)

    best_sim = -1.0
    best_shift = 0

    for shift in range(-MAX_SHIFT_BINS, MAX_SHIFT_BINS + 1):
        # figure out overlapping slice after shifting live relative to ref
        if shift >= 0:
            r = ref_roi[shift:]
            l = live_roi[:n - shift]
        else:
            r = ref_roi[:n + shift]
            l = live_roi[-shift:]

        if len(r) < 4:
            continue

        r_zm = r - np.mean(r)
        l_zm = l - np.mean(l)
        sim = cosine_similarity(l_zm, r_zm)

        if sim > best_sim:
            best_sim = sim
            best_shift = shift

    return best_sim >= GRASS_THRESHOLD, best_sim, best_shift


# ── User Input Thread ────────────────────────────────────────────
def input_thread():
    while not exit_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd in ('c', 'r'):
            capture_event.set()
        elif cmd in ('q', 'quit', 'exit'):
            exit_event.set()
            break
        time.sleep(0.05)


# ── Plot Styling ─────────────────────────────────────────────────
BG_DARK  = '#0f0f1a'
BG_PANEL = '#161625'
GRID_CLR = '#2a2a4a'
TEXT_CLR  = '#ccccdd'
ACCENT   = '#00d2ff'

def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(True, alpha=0.12, color=GRID_CLR)


# ── Main ─────────────────────────────────────────────────────────
def main():
    global grass_ref_db

    print("=" * 62)
    print("   xWRL1432 Grass Classifier — Range Profile Matching")
    print("=" * 62)
    print(f"   Range/bin:      {RANGE_PER_BIN*100:.2f} cm")
    print(f"   Analysis zone:  {NEAR_START_M} – {NEAR_END_M} m  "
          f"(bins {NEAR_BIN_START}–{NEAR_BIN_END})")
    print(f"   Threshold:      {GRASS_THRESHOLD}")
    print(f"   Height tol:     {HEIGHT_TOLERANCE_CM} cm ({MAX_SHIFT_BINS} bin max shift)")
    print()
    print("   'c' + Enter  →  record grass reference (point at grass first!)")
    print("   'r' + Enter  →  re-record reference")
    print("   'q' + Enter  →  quit")
    print("=" * 62)

    # ── Connect ──────────────────────────────────────────────────
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

    # ── Setup figure (3 panels: profile, classification, overlay) ─
    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28,
                           left=0.06, right=0.96, top=0.91, bottom=0.07)

    # Panel 1 (top-left): Live range profile
    ax_profile = fig.add_subplot(gs[0, 0])
    style_ax(ax_profile, 'Live Range Profile')
    ax_profile.set_xlabel('Range (m)', color=TEXT_CLR, fontsize=9)
    ax_profile.set_ylabel('Power (dB)', color=TEXT_CLR, fontsize=9)
    line_live, = ax_profile.plot([], [], color=ACCENT, linewidth=1.0, alpha=0.5,
                                  label='instantaneous')
    line_avg,  = ax_profile.plot([], [], color='#ffaa00', linewidth=2.0,
                                  label='averaged')
    ax_profile.axvspan(NEAR_START_M, NEAR_END_M, alpha=0.08, color='#00ff88',
                        label='ROI')
    ax_profile.legend(loc='upper right', fontsize=7, facecolor=BG_PANEL,
                      edgecolor=GRID_CLR, labelcolor='white')

    # Panel 2 (top-right): Classification result
    ax_result = fig.add_subplot(gs[0, 1])
    ax_result.set_facecolor(BG_PANEL)
    ax_result.set_xticks([])
    ax_result.set_yticks([])
    for spine in ax_result.spines.values():
        spine.set_color(GRID_CLR)
    result_text = ax_result.text(0.5, 0.55, 'No reference\nPress \'c\' to record grass',
                                  transform=ax_result.transAxes, ha='center', va='center',
                                  fontsize=20, color=TEXT_CLR, fontweight='bold')
    sim_text = ax_result.text(0.5, 0.18, '',
                               transform=ax_result.transAxes, ha='center', va='center',
                               fontsize=13, color=TEXT_CLR)
    ax_result.set_title('Classification', color='white', fontsize=12,
                         fontweight='bold', pad=10)

    # Panel 3 (bottom, full width): Profile overlay
    ax_compare = fig.add_subplot(gs[1, :])
    style_ax(ax_compare, 'Grass Reference vs Live (ROI)')
    ax_compare.set_xlabel('Range (m)', color=TEXT_CLR, fontsize=9)
    ax_compare.set_ylabel('Power (dB)', color=TEXT_CLR, fontsize=9)

    fig.suptitle('xWRL1432 Grass Classifier',
                 color='white', fontsize=15, fontweight='bold', y=0.97)
    plt.show(block=False)

    # ── Main Loop ────────────────────────────────────────────────
    avg_buffer = deque(maxlen=AVG_FRAMES)
    frame_idx = 0

    while not exit_event.is_set():
        with data_lock:
            profile = latest_profile.copy() if latest_profile is not None else None

        if profile is None:
            plt.pause(0.05)
            continue

        frame_idx += 1

        profile_db = 10 * np.log10(profile + 1)
        avg_buffer.append(profile_db)
        avg_profile = np.mean(avg_buffer, axis=0)

        # ── Capture grass reference ──────────────────────────────
        if capture_event.is_set():
            capture_event.clear()
            grass_ref_db = avg_profile.copy()
            print(f"  -> Grass reference recorded ({len(avg_buffer)} frames averaged)")

        # ── Panel 1: live profile ────────────────────────────────
        line_live.set_data(RANGE_AXIS, profile_db)
        line_avg.set_data(RANGE_AXIS, avg_profile)
        ax_profile.set_xlim(0, min(MAX_RANGE, 4.0))
        lo = max(0, np.percentile(avg_profile, 5) - 5)
        hi = np.max(avg_profile) + 10
        ax_profile.set_ylim(lo, hi)

        # ── Panel 2: classification result ───────────────────────
        if grass_ref_db is not None:
            is_grass, sim, shift = classify_grass(avg_profile, grass_ref_db)

            if is_grass:
                label = 'GRASS'
                color = '#2ecc71'
            else:
                label = 'NOT GRASS'
                color = '#e74c3c'

            result_text.set_text(label)
            result_text.set_color(color)
            result_text.set_fontsize(42)
            shift_cm = shift * RANGE_PER_BIN * 100
            sim_text.set_text(
                f'similarity: {sim:.3f}  (threshold: {GRASS_THRESHOLD})\n'
                f'shift: {shift:+d} bin ({shift_cm:+.1f} cm)')
            sim_text.set_color(TEXT_CLR)

        # ── Panel 3: overlay (every 5 frames) ───────────────────
        if frame_idx % 5 == 0:
            ax_compare.clear()
            style_ax(ax_compare, 'Grass Reference vs Live (ROI)')
            ax_compare.set_xlabel('Range (m)', color=TEXT_CLR, fontsize=9)
            ax_compare.set_ylabel('Power (dB)', color=TEXT_CLR, fontsize=9)

            near_range = RANGE_AXIS[NEAR_BIN_START:NEAR_BIN_END]
            cur_roi = avg_profile[NEAR_BIN_START:NEAR_BIN_END]

            if grass_ref_db is not None:
                ref_roi = grass_ref_db[NEAR_BIN_START:NEAR_BIN_END]
                ax_compare.plot(near_range, ref_roi,
                                color='#2ecc71', linewidth=2.2, label='grass ref', alpha=0.9)
                ax_compare.fill_between(near_range, ref_roi, cur_roi,
                                         alpha=0.15, color='#e74c3c', label='difference')

            ax_compare.plot(near_range, cur_roi,
                            color='white', linewidth=1.5, linestyle='--',
                            label='live', alpha=0.7)

            ax_compare.set_xlim(NEAR_START_M, NEAR_END_M)
            ax_compare.legend(loc='upper right', fontsize=8, facecolor=BG_PANEL,
                              edgecolor=GRID_CLR, labelcolor='white')

        # ── Refresh ──────────────────────────────────────────────
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except:
            break

        plt.pause(0.05)

    print("\nShutting down...")
    try:
        ser.close()
    except:
        pass


if __name__ == "__main__":
    main()