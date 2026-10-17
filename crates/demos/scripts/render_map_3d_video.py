#!/usr/bin/env python3
"""Minimal 3D map video: a camera flies around the point-cloud map while it is
being built, scan by scan. No UI, just the map (and a faint sensor path).

Input: the `--record` log of `unitree_l1_bag` (per-node scan points in the node
frame + optimized node poses). Every node is placed at its final optimized
pose, so the video shows the map Cartographer converges to, revealed in the
order the scans arrived.

Usage:
  python render_map_3d_video.py target/unitree_l1/slam_run.jsonl --out target/media/map_3d.mp4
Needs numpy, Pillow, matplotlib (colormap only) and `ffmpeg` on PATH.
"""

import argparse
import json
import subprocess

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageFilter


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose(p):
    return np.asarray(p[:3], float), quat_to_rot(p[3:7])


def load(path):
    nodes, final_poses, world_from_map = {}, {}, None
    for line in open(path):
        d = json.loads(line)
        if d["type"] == "node":
            nodes[d["id"]] = (d["t"], d["global"], np.asarray(d["points"], float).reshape(-1, 3))
        elif d["type"] == "optimization":
            for n in d["nodes"]:
                final_poses[n[0]] = n[1:8]
        elif d["type"] == "final":
            world_from_map = d["world_from_map"]
    t_wm, r_wm = pose(world_from_map)
    ids = sorted(nodes)
    clouds, path = [], []
    for i in ids:
        _, live, pts = nodes[i]
        t_mn, r_mn = pose(final_poses.get(i, live))
        t = r_wm @ t_mn + t_wm
        r = r_wm @ r_mn
        clouds.append(pts @ r.T + t)
        path.append(t)
    return clouds, np.array(path)


def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])  # rows: camera x (right), y (up), z (forward)


def render(points, colors, eye, target, w, h, fov_deg, splat):
    rot = look_at(eye, target)
    pc = (points - eye) @ rot.T
    keep = pc[:, 2] > 0.3
    pc, col = pc[keep], colors[keep]
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    u = (f * pc[:, 0] / pc[:, 2] + w / 2).astype(int)
    v = (-f * pc[:, 1] / pc[:, 2] + h / 2).astype(int)
    depth = pc[:, 2]
    # Depth fog: far points fade towards the background.
    fog = np.clip(1.15 - depth / 55.0, 0.35, 1.0)[:, None]
    col = col * fog
    order = np.argsort(-depth)  # far to near: near points are written last
    u, v, col = u[order], v[order], col[order]
    img = np.zeros((h, w, 3), np.float32)
    for du in range(splat):
        for dv in range(splat):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = col[ok]
    return img


def polyline(img, pts2d, color, width):
    from PIL import ImageDraw
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    if len(pts2d) > 1:
        d.line([tuple(p) for p in pts2d], fill=color, width=width, joint="curve")
    return np.asarray(pil).astype(np.float32) / 255


def project(points, eye, target, w, h, fov_deg):
    rot = look_at(eye, target)
    pc = (points - eye) @ rot.T
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return np.stack([f * pc[:, 0] / pc[:, 2] + w / 2, -f * pc[:, 1] / pc[:, 2] + h / 2], -1), pc[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--build-seconds", type=float, default=20.0)
    ap.add_argument("--hold-seconds", type=float, default=6.0)
    ap.add_argument("--ceiling", type=float, default=2.75, help="hide points above this height (m)")
    args = ap.parse_args()

    clouds, path = load(args.record)
    sizes = np.array([len(c) for c in clouds])
    starts = np.concatenate([[0], np.cumsum(sizes)])
    pts = np.concatenate(clouds)
    node_of_point = np.repeat(np.arange(len(clouds)), sizes)
    keep = pts[:, 2] < args.ceiling
    pts, node_of_point = pts[keep], node_of_point[keep]
    cmap = colormaps["turbo"]
    base = cmap(np.clip(pts[:, 2] / args.ceiling, 0, 1) * 0.9 + 0.05)[:, :3].astype(np.float32)
    center = np.array([0.0, 0.0, 0.9])

    ss = 2  # supersampling factor for anti-aliasing
    W, H = args.width * ss, args.height * ss
    n_build = int(args.build_seconds * args.fps)
    n_total = n_build + int(args.hold_seconds * args.fps)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
         "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)
    n_nodes = len(clouds)
    for k in range(n_total):
        s = min(k / max(n_build - 1, 1), 1.0)
        upto = int(round(s * (n_nodes - 1)))
        # Camera: one smooth orbit over the whole video, slowly descending.
        a = np.radians(-100 + 360 * k / n_total)
        elev = np.radians(42 - 12 * (k / n_total))
        radius = 27.0 - 3.5 * np.sin(np.pi * k / n_total)
        eye = center + radius * np.array([np.cos(a) * np.cos(elev), np.sin(a) * np.cos(elev), np.sin(elev)])

        vis = node_of_point <= upto
        col = base[vis].copy()
        # Freshly added scans glow briefly.
        age = upto - node_of_point[vis]
        glow = np.clip(1 - age / 25.0, 0, 1)[:, None]
        col = col * (1 - 0.6 * glow) + 0.6 * glow
        img = render(pts[vis], col, eye, center, W, H, 50, splat=3)

        # Faint sensor path and current position.
        p2, d = project(path[: upto + 1], eye, center, W, H, 50)
        p2 = p2[d > 0.3]
        if len(p2) > 1:
            trail = polyline(np.zeros_like(img), p2, (255, 255, 255), 2 * ss)
            img = np.maximum(img, 0.45 * trail)
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        if len(p2) and upto < n_nodes - 1:
            from PIL import ImageDraw
            dr = ImageDraw.Draw(pil)
            x, y = p2[-1]
            r = 7 * ss
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
        pil = pil.filter(ImageFilter.GaussianBlur(0.6)).resize((args.width, args.height), Image.LANCZOS)
        ff.stdin.write(pil.tobytes())
        if k % 60 == 0:
            print(f"frame {k}/{n_total}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
