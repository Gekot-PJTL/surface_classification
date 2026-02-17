#!/usr/bin/env python3
"""
Surface Classification via Range Profile — xWRL1432

Captures range profiles and computes features that distinguish surface types:
  - Mean reflected power (hard surfaces reflect more)
  - Peak-to-mean ratio (smooth surfaces have sharper peaks)
  - Bin-to-bin variance (rough surfaces scatter more diffusely)
  - Decay rate (how quickly signal falls off with range)

Usage:
  1. Run the script, point radar at a surface
  2. Type 'c' + Enter in terminal to capture a snapshot
  3. Enter a name (e.g. 'grass', 'concrete', 'gravel')
  4. Move to another surface, repeat
  5. Compare the profiles and features across panels
  6. Type 'q' to quit
"""

import serial
import time
import struct
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# ── Radar Config ──────────────────────────────────────────────────
PORT_CLI       = 'COM12'
BAUD_INITIAL   = 115200
BAUD_DATA      = 1250000
CFG_FILE       = 'surface_range.cfg'

MAGIC_WORD         = b'\x02\x01\x04\x03\x06\x05\x08\x07'
PACKET_HEADER_SIZE = 40
TLV_HEADER_SIZE    = 8
TLV_RANGE_PROFILE  = 302   # Extended msg format
TLV_RANGE_PROFILE_STD = 2  # Standard msg format

# ── Radar Parameters (derived from .cfg) ─────────────────────────
NUM_ADC_SAMPLES  = 256
SAMPLE_RATE_MHZ  = 12.5        # 100 / DigOutputSampRate(8)
SLOPE_MHZ_PER_US = 160.0
C                = 3e8

NUM_RANGE_BINS   = NUM_ADC_SAMPLES // 2   # 128
RANGE_PER_BIN    = (C * SAMPLE_RATE_MHZ * 1e6) / \
                   (2 * SLOPE_MHZ_PER_US * 1e12 * NUM_ADC_SAMPLES)
MAX_RANGE        = RANGE_PER_BIN * NUM_RANGE_BINS
RANGE_AXIS       = np.arange(NUM_RANGE_BINS) * RANGE_PER_BIN

# Analysis zone: skip DC bin, focus on near field (ground reflection)
NEAR_START_M = 0.1
NEAR_END_M   = 2.0
NEAR_BIN_START = max(1, int(NEAR_START_M / RANGE_PER_BIN))
NEAR_BIN_END   = min(NUM_RANGE_BINS, int(NEAR_END_M / RANGE_PER_BIN))

# ── Shared State ─────────────────────────────────────────────────
data_lock          = threading.Lock()
latest_profile     = None
exit_event         = threading.Event()
capture_event      = threading.Event()
captured_surfaces  = {}  # name -> {'profile': array, 'features': dict}

SURFACE_COLORS = [
    '#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c',
    '#e67e22', '#e84393', '#00cec9', '#fd79a8',
]


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


# ── Feature Extraction ───────────────────────────────────────────
def compute_features(profile_linear):
    """
    Compute surface classification features from near-field range bins.
    Input is linear-scale power values.
    """
    near = profile_linear[NEAR_BIN_START:NEAR_BIN_END].copy()

    if len(near) == 0 or np.max(near) == 0:
        return {'mean_power': 0, 'peak_to_mean': 0, 'variance': 0, 'decay_rate': 0}

    near_db = 10 * np.log10(near + 1)

    # 1. Mean reflected power in analysis zone (dB)
    mean_power = float(np.mean(near_db))

    # 2. Peak-to-mean ratio — high for specular (smooth) surfaces
    peak_to_mean = float(np.max(near_db) / (mean_power + 1e-6))

    # 3. Bin-to-bin variance — high for rough/scattering surfaces
    diffs = np.diff(near_db)
    variance = float(np.var(diffs))

    # 4. Decay rate — linear fit slope of dB vs range bin
    x = np.arange(len(near_db))
    if len(x) > 1:
        coeffs = np.polyfit(x, near_db, 1)
        decay_rate = float(-coeffs[0])  # positive = decaying
    else:
        decay_rate = 0.0

    return {
        'mean_power':   round(mean_power, 2),
        'peak_to_mean': round(peak_to_mean, 3),
        'variance':     round(variance, 3),
        'decay_rate':   round(decay_rate, 4),
    }


# ── User Input Thread ────────────────────────────────────────────
def input_thread():
    while not exit_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd == 'c':
            capture_event.set()
        elif cmd in ('q', 'quit', 'exit'):
            exit_event.set()
            break
        time.sleep(0.05)


# ── Plot Styling ─────────────────────────────────────────────────
BG_DARK   = '#0f0f1a'
BG_PANEL  = '#161625'
GRID_CLR  = '#2a2a4a'
TEXT_CLR   = '#ccccdd'
ACCENT    = '#00d2ff'

def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(True, alpha=0.12, color=GRID_CLR)


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("   xWRL1432 Surface Classifier — Range Profile Analysis")
    print("=" * 62)
    print(f"   Range/bin:      {RANGE_PER_BIN*100:.2f} cm")
    print(f"   Max range:      {MAX_RANGE:.2f} m")
    print(f"   Analysis zone:  {NEAR_START_M} – {NEAR_END_M} m  "
          f"(bins {NEAR_BIN_START}–{NEAR_BIN_END})")
    print()
    print("   'c' + Enter  →  capture current surface")
    print("   'q' + Enter  →  quit")
    print("=" * 62)

    # ── Connect ──────────────────────────────────────────────────
    ser, _ = try_open_at_correct_baud()
    time.sleep(0.2)
    ser = send_config(ser, CFG_FILE)

    threading.Thread(target=read_data_stream, args=(ser,), daemon=True).start()
    threading.Thread(target=input_thread, daemon=True).start()

    # Wait for first data
    print("  Waiting for data...", end='', flush=True)
    for _ in range(100):
        with data_lock:
            if latest_profile is not None:
                break
        time.sleep(0.1)
    print(" OK\n")

    # ── Setup figure ─────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28,
                           left=0.06, right=0.96, top=0.91, bottom=0.07)

    # -- Panel 1: Live range profile -----
    ax_profile = fig.add_subplot(gs[0, 0])
    style_ax(ax_profile, 'Live Range Profile')
    ax_profile.set_xlabel('Range (m)', color=TEXT_CLR, fontsize=9)
    ax_profile.set_ylabel('Power (dB)', color=TEXT_CLR, fontsize=9)
    line_live, = ax_profile.plot([], [], color=ACCENT, linewidth=1.0, alpha=0.5,
                                  label='instantaneous')
    line_avg,  = ax_profile.plot([], [], color='#ffaa00', linewidth=2.0,
                                  label='averaged (20 frames)')
    ax_profile.axvspan(NEAR_START_M, NEAR_END_M, alpha=0.08, color='#00ff88',
                        label='analysis zone')
    ax_profile.legend(loc='upper right', fontsize=7, facecolor=BG_PANEL,
                      edgecolor=GRID_CLR, labelcolor='white')

    # -- Panel 2: Feature time series -----
    ax_feat = fig.add_subplot(gs[0, 1])
    style_ax(ax_feat, 'Surface Features (Rolling)')
    ax_feat.set_xlabel('Frame', color=TEXT_CLR, fontsize=9)

    feat_keys    = ['mean_power', 'peak_to_mean', 'variance', 'decay_rate']
    feat_labels  = ['Mean Power (dB)', 'Peak / Mean', 'Bin Variance', 'Decay Rate']
    feat_colors  = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    feat_history = {k: deque(maxlen=300) for k in feat_keys}
    feat_lines   = {}

    # Use twin axes so different-scaled features are readable
    ax_feat_r = ax_feat.twinx()
    ax_feat_r.tick_params(colors=TEXT_CLR, labelsize=8)
    ax_feat_r.spines['right'].set_color(GRID_CLR)

    for i, (key, label, color) in enumerate(zip(feat_keys, feat_labels, feat_colors)):
        target_ax = ax_feat if i == 0 else ax_feat_r
        ln, = target_ax.plot([], [], color=color, linewidth=1.4, label=label, alpha=0.85)
        feat_lines[key] = (ln, target_ax)

    # Combined legend
    lines_all = [feat_lines[k][0] for k in feat_keys]
    ax_feat.legend(lines_all, feat_labels, loc='upper left', fontsize=6,
                   facecolor=BG_PANEL, edgecolor=GRID_CLR, labelcolor='white', ncol=2)

    # -- Panel 3: Captured profiles overlay -----
    ax_compare = fig.add_subplot(gs[1, 0])
    style_ax(ax_compare, 'Captured Surface Profiles (press \'c\' to capture)')
    ax_compare.set_xlabel('Range (m)', color=TEXT_CLR, fontsize=9)
    ax_compare.set_ylabel('Power (dB)', color=TEXT_CLR, fontsize=9)

    # -- Panel 4: Feature bar comparison -----
    ax_bars = fig.add_subplot(gs[1, 1])
    style_ax(ax_bars, 'Feature Comparison')

    fig.suptitle('xWRL1432 Surface Classification Dashboard',
                 color='white', fontsize=15, fontweight='bold', y=0.97)
    plt.show(block=False)

    # ── Main Loop ────────────────────────────────────────────────
    avg_buffer = deque(maxlen=20)
    frame_idx = 0
    capture_count = 0

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

        features = compute_features(profile)
        for key in feat_history:
            feat_history[key].append(features[key])

        # ── Capture ──────────────────────────────────────────────
        if capture_event.is_set():
            capture_event.clear()
            capture_count += 1
            name = input(f"  Surface name (e.g. grass, concrete): ").strip()
            if not name:
                name = f"surface_{capture_count}"

            avg_linear = np.mean([10 ** (p / 10) - 1 for p in avg_buffer], axis=0)
            avg_linear = np.clip(avg_linear, 0, None)

            captured_surfaces[name] = {
                'profile_db': avg_profile.copy(),
                'features': compute_features(avg_linear),
            }
            print(f"  -> Captured '{name}': {captured_surfaces[name]['features']}")
            print(f"     ({len(captured_surfaces)} surface(s) stored)\n")

        # ── Panel 1 Update ───────────────────────────────────────
        line_live.set_data(RANGE_AXIS, profile_db)
        line_avg.set_data(RANGE_AXIS, avg_profile)
        ax_profile.set_xlim(0, min(MAX_RANGE, 4.0))
        lo = max(0, np.percentile(avg_profile, 5) - 5)
        hi = np.max(avg_profile) + 10
        ax_profile.set_ylim(lo, hi)

        # ── Panel 2 Update ───────────────────────────────────────
        for key, (ln, target_ax) in feat_lines.items():
            data = list(feat_history[key])
            ln.set_data(range(len(data)), data)
        x_max = max(300, frame_idx)
        ax_feat.set_xlim(max(0, frame_idx - 300), frame_idx)

        # Rescale left axis (mean_power)
        mp = list(feat_history['mean_power'])
        if mp:
            ax_feat.set_ylim(max(0, min(mp[-100:]) - 2), max(mp[-100:]) + 2)

        # Rescale right axis (other features)
        right_vals = []
        for k in ['peak_to_mean', 'variance', 'decay_rate']:
            right_vals.extend(list(feat_history[k])[-100:])
        if right_vals:
            ax_feat_r.set_ylim(max(0, min(right_vals) - 0.5), max(right_vals) + 0.5)

        # ── Panel 3 Update (every 10 frames) ─────────────────────
        if captured_surfaces and frame_idx % 10 == 0:
            ax_compare.clear()
            style_ax(ax_compare, 'Captured Surface Profiles')
            ax_compare.set_xlabel('Range (m)', color=TEXT_CLR, fontsize=9)
            ax_compare.set_ylabel('Power (dB)', color=TEXT_CLR, fontsize=9)

            near_range = RANGE_AXIS[NEAR_BIN_START:NEAR_BIN_END]

            for i, (name, data) in enumerate(captured_surfaces.items()):
                color = SURFACE_COLORS[i % len(SURFACE_COLORS)]
                near_db = data['profile_db'][NEAR_BIN_START:NEAR_BIN_END]
                ax_compare.plot(near_range, near_db,
                                color=color, linewidth=2.2, label=name, alpha=0.9)

            # Current (dashed)
            cur_near = avg_profile[NEAR_BIN_START:NEAR_BIN_END]
            ax_compare.plot(near_range, cur_near,
                            color='white', linewidth=1.2, linestyle='--',
                            label='live', alpha=0.45)

            ax_compare.set_xlim(NEAR_START_M, NEAR_END_M)
            ax_compare.legend(loc='upper right', fontsize=8, facecolor=BG_PANEL,
                              edgecolor=GRID_CLR, labelcolor='white')

        # ── Panel 4 Update (every 10 frames) ─────────────────────
        if captured_surfaces and frame_idx % 10 == 0:
            ax_bars.clear()
            style_ax(ax_bars, 'Feature Comparison')

            n_surfaces = len(captured_surfaces)
            n_feats = len(feat_keys)
            bar_w = 0.75 / max(n_surfaces + 1, 2)
            x_pos = np.arange(n_feats)

            for i, (name, data) in enumerate(captured_surfaces.items()):
                color = SURFACE_COLORS[i % len(SURFACE_COLORS)]
                vals = [data['features'][k] for k in feat_keys]
                ax_bars.bar(x_pos + i * bar_w, vals, bar_w,
                            label=name, color=color, alpha=0.85,
                            edgecolor='white', linewidth=0.4)

            # Current live
            cur_vals = [features[k] for k in feat_keys]
            ax_bars.bar(x_pos + n_surfaces * bar_w, cur_vals, bar_w,
                        label='live', color='white', alpha=0.3,
                        edgecolor='white', linewidth=0.4)

            short_labels = ['Mean\nPower', 'Peak /\nMean', 'Bin\nVariance', 'Decay\nRate']
            ax_bars.set_xticks(x_pos + bar_w * n_surfaces / 2)
            ax_bars.set_xticklabels(short_labels, fontsize=8, color=TEXT_CLR)
            ax_bars.legend(loc='upper right', fontsize=7, facecolor=BG_PANEL,
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
