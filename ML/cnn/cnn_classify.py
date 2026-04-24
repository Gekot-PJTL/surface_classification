#!/usr/bin/env python3
"""
Grass Classifier — Live Inference with Trained Model

Loads grass_model.pth (trained by grass_train.py) and classifies
the live radar range profile in real time.

Usage:
    python grass_classify.py
    python grass_classify.py --model my_model.pth --threshold 0.5
"""

import os
import serial
import time
import struct
import threading
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# ── Model (must match grass_train.py) ────────────────────────────
class GrassNet(nn.Module):
    def __init__(self, n_bins):
        super().__init__()
        self.n_bins = n_bins
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Radar Config ──────────────────────────────────────────────────
PORT_CLI       = 'COM12'
BAUD_INITIAL   = 115200
CFG_FILE       = os.path.join(os.path.dirname(__file__), '..', 'surface_range.cfg')

MAGIC_WORD         = b'\x02\x01\x04\x03\x06\x05\x08\x07'
PACKET_HEADER_SIZE = 40
TLV_HEADER_SIZE    = 8
TLV_RANGE_PROFILE  = 302
TLV_RANGE_PROFILE_STD = 2

NUM_ADC_SAMPLES  = 256
SAMPLE_RATE_MHZ  = 12.5
SLOPE_MHZ_PER_US = 160.0
C                = 3e8

NUM_RANGE_BINS   = NUM_ADC_SAMPLES // 2
RANGE_PER_BIN    = (C * SAMPLE_RATE_MHZ * 1e6) / \
                   (2 * SLOPE_MHZ_PER_US * 1e12 * NUM_ADC_SAMPLES)
MAX_RANGE        = RANGE_PER_BIN * NUM_RANGE_BINS
RANGE_AXIS       = np.arange(NUM_RANGE_BINS) * RANGE_PER_BIN

NEAR_START_M   = 0.75
NEAR_END_M     = 3.8
NEAR_BIN_START = max(1, int(NEAR_START_M / RANGE_PER_BIN))
NEAR_BIN_END   = min(NUM_RANGE_BINS, int(NEAR_END_M / RANGE_PER_BIN))

AVG_FRAMES = 10

# ── Shared State ─────────────────────────────────────────────────
data_lock      = threading.Lock()
latest_profile = None
exit_event     = threading.Event()


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
    while not exit_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd in ('q', 'quit', 'exit'):
            exit_event.set()
            break
        time.sleep(0.05)


# ── ML Classification ───────────────────────────────────────────
def classify_grass_ml(model, live_db, threshold):
    """Run the trained model on the ROI and return (is_grass, probability)."""
    roi = live_db[NEAR_BIN_START:NEAR_BIN_END].astype(np.float32)

    # Same per-sample normalization used during training
    roi = (roi - roi.mean()) / (roi.std() + 1e-8)

    x = torch.tensor(roi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, bins)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    return prob >= threshold, prob


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
    parser = argparse.ArgumentParser(description="Live grass classifier")
    parser.add_argument("--model", default="cnn_model.pth", help="Trained model path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (0-1, higher = stricter grass)")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    n_bins = checkpoint["n_bins"]
    val_acc = checkpoint.get("best_val_acc", "?")

    expected_bins = NEAR_BIN_END - NEAR_BIN_START
    if n_bins != expected_bins:
        print(f"WARNING: Model trained on {n_bins} bins but ROI has {expected_bins} bins.")
        print(f"  Check NEAR_START_M / NEAR_END_M match between logger and classifier.")

    model = GrassNet(n_bins)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded ({n_bins} bins, val_acc={val_acc})")

    # ── Banner ───────────────────────────────────────────────────
    print("=" * 62)
    print("   xWRL1432 Grass Classifier — ML Inference")
    print("=" * 62)
    print(f"   ROI:        {NEAR_START_M}–{NEAR_END_M} m  ({expected_bins} bins)")
    print(f"   Threshold:  {args.threshold}")
    print(f"   Avg window: {AVG_FRAMES} frames")
    print()
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

    # ── Setup figure ─────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28,
                           left=0.06, right=0.96, top=0.91, bottom=0.07)

    # Panel 1: Live range profile
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

    # Panel 2: Classification result
    ax_result = fig.add_subplot(gs[0, 1])
    ax_result.set_facecolor(BG_PANEL)
    ax_result.set_xticks([])
    ax_result.set_yticks([])
    for spine in ax_result.spines.values():
        spine.set_color(GRID_CLR)
    result_text = ax_result.text(0.5, 0.55, 'Classifying...',
                                  transform=ax_result.transAxes, ha='center', va='center',
                                  fontsize=20, color=TEXT_CLR, fontweight='bold')
    prob_text = ax_result.text(0.5, 0.18, '',
                               transform=ax_result.transAxes, ha='center', va='center',
                               fontsize=13, color=TEXT_CLR)
    ax_result.set_title('Classification', color='white', fontsize=12,
                         fontweight='bold', pad=10)

    # Panel 3: Probability history
    ax_history = fig.add_subplot(gs[1, :])
    style_ax(ax_history, 'Confidence History')
    ax_history.set_xlabel('Frame', color=TEXT_CLR, fontsize=9)
    ax_history.set_ylabel('P(grass)', color=TEXT_CLR, fontsize=9)
    ax_history.set_ylim(-0.05, 1.05)
    ax_history.axhline(y=args.threshold, color='#ff6b6b', linestyle='--',
                        alpha=0.5, label=f'threshold={args.threshold}')
    ax_history.legend(loc='upper right', fontsize=7, facecolor=BG_PANEL,
                      edgecolor=GRID_CLR, labelcolor='white')

    fig.suptitle('xWRL1432 Grass Classifier — ML',
                 color='white', fontsize=15, fontweight='bold', y=0.97)
    plt.show(block=False)

    # ── Main Loop ────────────────────────────────────────────────
    avg_buffer = deque(maxlen=AVG_FRAMES)
    prob_history = deque(maxlen=200)
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

        # ── Classify ─────────────────────────────────────────────
        is_grass, prob = classify_grass_ml(model, avg_profile, args.threshold)
        prob_history.append(prob)

        # ── Panel 1: live profile ────────────────────────────────
        line_live.set_data(RANGE_AXIS, profile_db)
        line_avg.set_data(RANGE_AXIS, avg_profile)
        ax_profile.set_xlim(0, min(MAX_RANGE, 4.0))
        lo = max(0, np.percentile(avg_profile, 5) - 5)
        hi = np.max(avg_profile) + 10
        ax_profile.set_ylim(lo, hi)

        # ── Panel 2: result ──────────────────────────────────────
        if is_grass:
            label = 'GRASS'
            color = '#2ecc71'
        else:
            label = 'NOT GRASS'
            color = '#e74c3c'

        result_text.set_text(label)
        result_text.set_color(color)
        result_text.set_fontsize(42)
        prob_text.set_text(f'P(grass) = {prob:.3f}  (threshold: {args.threshold})')
        prob_text.set_color(TEXT_CLR)

        # ── Panel 3: history (every 5 frames) ────────────────────
        if frame_idx % 5 == 0 and len(prob_history) > 1:
            ax_history.clear()
            style_ax(ax_history, 'Confidence History')
            ax_history.set_xlabel('Frame', color=TEXT_CLR, fontsize=9)
            ax_history.set_ylabel('P(grass)', color=TEXT_CLR, fontsize=9)
            ax_history.set_ylim(-0.05, 1.05)
            ax_history.axhline(y=args.threshold, color='#ff6b6b', linestyle='--', alpha=0.5)

            hist = list(prob_history)
            xs = list(range(len(hist)))
            colors_fill = ['#2ecc7140' if p >= args.threshold else '#e74c3c40' for p in hist]
            ax_history.bar(xs, hist, color=colors_fill, width=1.0)
            ax_history.plot(xs, hist, color='white', linewidth=1.2, alpha=0.8)

        # ── Refresh ──────────────────────────────────────────────
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            break

        plt.pause(0.05)

    print("\nShutting down...")
    try:
        ser.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
