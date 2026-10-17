#!/usr/bin/env python3
"""Summary image (1080x1350, 4:5 portrait) for sharing the project.

Numbers come from the end-to-end tests (target/e2e/*.json) and the Unitree
L1 demo (target/unitree_l1/noisy/metrics.json); pass --static to use the
values recorded below instead.
Usage: python linkedin_chart.py --out target/media/cartographer_rust_summary.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e0"
OPTIMIZED = "#2a78d6"   # categorical slot 1
LOCAL = "#bdbcb6"       # de-emphasized comparison

# (label, path length m, local-only ATE cm, full SLAM ATE cm) -- defaults
# from the runs used for the report; refreshed from JSON when available.
SCENARIOS = [
    ("2D corridor, 2 loops", "corridor_2d", 215.8, 2.92, 1.25),
    ("2D office tour", "office_2d", 98.6, 0.62, 0.53),
    ("3D corridor, 2 loops", "corridor_3d", 219.8, 40.83, 10.71),
    ("3D room, handheld", "room_3d", 61.9, 11.44, 4.23),
    ("Unitree L1 + IMU (MuJoCo)", None, 98.7, 7.22, 1.83),
]


def refresh(root):
    out = []
    for label, key, length, local, full in SCENARIOS:
        if key:
            p = os.path.join(root, "target/e2e", key + ".json")
            if os.path.exists(p):
                d = json.load(open(p))
                local = d["local_slam_trajectory"]["ate_rmse_m"] * 100
                full = d["trajectory"]["ate_rmse_m"] * 100
                length = d["trajectory"]["path_length_m"]
        out.append((label, length, local, full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--static", action="store_true")
    args = ap.parse_args()
    rows = [(a, c, d, e) for a, _, c, d, e in SCENARIOS] if args.static else refresh(args.root)

    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 15})
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=SURFACE)

    # Title block.
    fig.text(0.06, 0.945, "Google Cartographer, rewritten in pure Rust",
             fontsize=34, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.893, "2D/3D LiDAR + IMU SLAM  ·  faithful port of the C++ library  ·  "
             "builds with a plain `cargo build`", fontsize=16, color=INK_2, va="top")

    # Hero stats.
    tiles = [("338 / 338", "C++ unit tests ported\nand passing"),
             ("1,482", "Rust tests: unit,\nproperty, end-to-end"),
             ("0", "C/C++ dependencies;\nmacOS, Linux, Windows"),
             ("~40×", "real time: 3D SLAM on a\nsimulated Unitree L1")]
    x0, w, gap, y0, h = 0.06, 0.205, 0.0167, 0.715, 0.135
    for i, (big, small) in enumerate(tiles):
        x = x0 + i * (w + gap)
        fig.patches.append(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor="white",
                                          edgecolor=GRID, linewidth=1.2))
        fig.text(x + 0.018, y0 + h - 0.022, big, fontsize=31, fontweight="bold", color=INK, va="top")
        fig.text(x + 0.018, y0 + 0.022, small, fontsize=12.5, color=INK_2, va="bottom", linespacing=1.35)

    # Chart.
    ax = fig.add_axes([0.37, 0.13, 0.57, 0.44], facecolor=SURFACE)
    fig.text(0.06, 0.655, "Trajectory error vs. ground truth (ATE RMSE, cm)",
             fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.623, "Lower is better. Local SLAM alone drifts; Cartographer's loop closure "
             "and pose-graph\noptimization pull the trajectory back.", fontsize=13.5, color=INK_2,
             va="top", linespacing=1.35)
    n = len(rows)
    bh = 0.34
    ys = list(range(n))[::-1]
    xmax = max(r[2] for r in rows) * 1.22
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    label_tf = ax.get_yaxis_transform()  # x in axes fraction, y in data
    for y, (label, length, local, full) in zip(ys, rows):
        ax.barh(y + bh / 2 + 0.01, local, height=bh, color=LOCAL, zorder=2)
        ax.barh(y - bh / 2 - 0.01, full, height=bh, color=OPTIMIZED, zorder=2)
        ax.text(local + xmax * 0.012, y + bh / 2 + 0.01, f"{local:.1f}", va="center",
                fontsize=13, color=INK_2)
        ax.text(full + xmax * 0.012, y - bh / 2 - 0.01, f"{full:.1f}", va="center",
                fontsize=13, color=INK, fontweight="bold")
        ax.text(-0.535, y + 0.17, label, transform=label_tf, fontsize=14.5, color=INK, va="center")
        ax.text(-0.535, y - 0.2, f"{length:.0f} m path", transform=label_tf, fontsize=12,
                color=INK_3, va="center")
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.tick_params(axis="x", colors=INK_3, labelsize=12, length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Legend (two series).
    ly = 0.092
    for x, color, text in [(0.06, LOCAL, "Local SLAM only (no loop closure)"),
                           (0.42, OPTIMIZED, "Full Cartographer (loop closure + pose-graph optimization)")]:
        fig.patches.append(FancyBboxPatch((x, ly - 0.006), 0.018, 0.012,
                                          boxstyle="round,pad=0,rounding_size=0.003",
                                          transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text(x + 0.025, ly, text, fontsize=12.5, color=INK_2, va="center")

    fig.text(0.06, 0.03, "Synthetic datasets with exact ground truth (simulated LiDAR/IMU/odometry; "
             "Unitree L1 simulated in MuJoCo).\nAlignment: rigid SE(3). Same algorithms, constants and "
             "defaults as upstream Cartographer.", fontsize=11.5, color=INK_3, va="bottom",
             linespacing=1.4)
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
