#!/usr/bin/env python3
"""Summary image (1080x1350, 4:5 portrait) for sharing the project: a rendered
view of the 3D map plus four plain-language numbers. Dark theme, matching
the map videos.

Usage:
  python linkedin_chart.py target/unitree_l1/slam_run.jsonl --out target/media/cartographer_rust_summary.png
"""
import argparse
import json
import os
import sys

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_map_3d_video import load, polyline, project, render  # noqa: E402

W, H = 1080, 1350
BG = (10, 12, 18)
INK = (240, 243, 248)
INK_2 = (150, 158, 172)
ACCENT = (232, 116, 59)  # rust orange, matching the warm wall tops of the map
CARD = (22, 27, 38)


def font(size, bold=False):
    candidates = (["/System/Library/Fonts/HelveticaNeue.ttc"] if sys.platform == "darwin" else []) + [
        "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def map_render(record, width, height):
    clouds, path = load(record)
    pts = np.concatenate(clouds)
    pts = pts[pts[:, 2] < 2.75]
    col = colormaps["turbo"](np.clip(pts[:, 2] / 2.75, 0, 1) * 0.9 + 0.05)[:, :3].astype(np.float32)
    ss = 2
    center = np.array([0.0, 0.0, 0.7])
    a, elev, radius = np.radians(-62), np.radians(38), 25.5
    eye = center + radius * np.array([np.cos(a) * np.cos(elev), np.sin(a) * np.cos(elev), np.sin(elev)])
    img = render(pts, col, eye, center, width * ss, height * ss, 46, splat=3)
    p2, d = project(path, eye, center, width * ss, height * ss, 46)
    trail = polyline(np.zeros_like(img), p2[d > 0.3], (255, 255, 255), 2 * ss)
    img = np.maximum(img, 0.4 * trail)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return pil.filter(ImageFilter.GaussianBlur(0.6)).resize((width, height), Image.LANCZOS)


def stat_list(final):
    ate_cm = final["ate_rmse_optimized_m"] * 100
    return [
        ("338 / 338", "of the original C++ tests\nported and passing"),
        (f"{ate_cm:.1f} cm", "average position error\nafter a 99 m walk"),
        ("2.4 s", "to map 90 s of sensor data\n(Apple M3 Max laptop)"),
        ("0", "C/C++ dependencies:\njust `cargo build`"),
    ]


def landscape(record, final, out):
    """1920x1080 (16:9): matches the videos, for use as a video thumbnail."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), BG)
    map_w = 1180
    hero = map_render(record, map_w, h)
    # Fade the map into the background on its left edge.
    fade = Image.new("L", (map_w, h), 255)
    fd = ImageDraw.Draw(fade)
    for i in range(220):
        fd.line([(i, 0), (i, h)], fill=int(255 * (i / 220) ** 1.5))
    img.paste(Image.composite(hero, Image.new("RGB", (map_w, h), BG), fade), (w - map_w, 0))
    d = ImageDraw.Draw(img)
    x0 = 96
    d.text((x0, 190), "Google Cartographer,", font=font(76, True), fill=INK)
    d.text((x0, 276), "rewritten in pure Rust", font=font(76, True), fill=ACCENT)
    cw, ch, top = 420, 210, 480
    for i, (big, small) in enumerate(stat_list(final)):
        cx = x0 + (i % 2) * cw
        cy = top + (i // 2) * ch
        d.text((cx, cy), big, font=font(66, True), fill=INK)
        d.multiline_text((cx, cy + 88), small, font=font(25), fill=INK_2, spacing=7)
    img.save(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", help="slam_run.jsonl written by unitree_l1_bag --record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--landscape", action="store_true",
                    help="1920x1080 (16:9) layout, e.g. as a LinkedIn video thumbnail")
    args = ap.parse_args()
    final = [json.loads(l) for l in open(args.record) if '"type":"final"' in l][0]
    if args.landscape:
        landscape(args.record, final, args.out)
        return

    img = Image.new("RGB", (W, H), BG)
    hero_h = 610
    img.paste(map_render(args.record, W, hero_h), (0, 250))
    # Fade the hero image into the background at top and bottom.
    fade = Image.new("L", (W, hero_h), 255)
    fd = ImageDraw.Draw(fade)
    for i in range(70):
        a = int(255 * i / 70)
        fd.line([(0, i), (W, i)], fill=a)
        fd.line([(0, hero_h - 1 - i), (W, hero_h - 1 - i)], fill=a)
    bg = Image.new("RGB", (W, hero_h), BG)
    img.paste(Image.composite(img.crop((0, 250, W, 250 + hero_h)), bg, fade), (0, 250))

    d = ImageDraw.Draw(img)
    x0 = 64
    d.text((x0, 70), "Google Cartographer,", font=font(58, True), fill=INK)
    d.text((x0, 136), "rewritten in pure Rust", font=font(58, True), fill=ACCENT)
    d.text((x0, 214), "LiDAR + IMU SLAM: a 3D map built from a simulated Unitree L1 scan",
           font=font(23), fill=INK_2)

    stats = stat_list(final)
    cw, ch, gap = 462, 170, 28
    top = 880
    for i, (big, small) in enumerate(stats):
        cx = x0 + (i % 2) * (cw + gap)
        cy = top + (i // 2) * (ch + gap)
        d.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=18, fill=CARD)
        d.text((cx + 30, cy + 22), big, font=font(56, True), fill=INK)
        d.multiline_text((cx + 30, cy + 98), small, font=font(22), fill=INK_2, spacing=6)

    d.text((x0, H - 52), "Simulated sensors with exact ground truth  ·  same algorithms & defaults as upstream",
           font=font(18), fill=(95, 102, 116))
    img.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
