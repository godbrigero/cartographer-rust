#!/usr/bin/env python3
"""Renders a video of a Cartographer run recorded with
`unitree_l1_bag --record FILE` (JSON lines).

    python render_slam_video.py target/unitree_l1/slam_run.jsonl \
        -o target/unitree_l1/slam_run.mp4 [--keyframes DIR] [--jobs N]

Needs numpy, matplotlib and ffmpeg on PATH.

Layout (1600x900, dark theme):
  * left: top-down view of the room. Each node's subsampled scan (ceiling
    removed, coloured by height) is placed at the node's current best pose
    estimate: the latest global optimization for nodes it covers, otherwise
    the live estimate (local pose mapped with the local-to-global transform
    known when the node was created). Ground truth (green), the current
    estimate of the trajectory (red), local SLAM only (blue, dashed) and the
    sensor pose. After each global optimization the map is animated from the
    old to the new poses, the loop-closure constraints flash (node to
    submap origin) and a banner shows the optimization number.
  * right top: rotating 3D view of the map and trajectory.
  * right middle: translation error vs time (live estimate and local SLAM
    only vs ground truth, plus the RMSE of the whole trajectory so far as
    currently estimated).
  * right bottom: statistics.

Everything is shown in the ground-truth (world) frame using the rigid
alignments computed by the example at the end of the run (`world_from_map`
for the SLAM estimates, `world_from_local` for local SLAM only): only a
fixed frame change, the estimates themselves are the ones the system had at
each moment.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

WIDTH, HEIGHT, DPI = 1600, 900, 100
FPS = 30

BG = "#0d1117"
PANEL = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
GT_COLOR = "#3fb950"
EST_COLOR = "#ff5f56"
LOCAL_COLOR = "#58a6ff"
BEST_COLOR = "#f0a830"
LOOP_COLOR = "#ff6ef5"
INTER_COLOR = "#6ad7ff"
SUBMAP_COLOR = "#ffd84d"
CMAP = colormaps["turbo"]

# Animation of an optimization: frames morphing old -> new poses, then
# frames holding the result with the constraints shown.
MORPH_FRAMES = 10
HOLD_FRAMES = 14
HOLD_FRAMES_NO_LOOP = 6
END_FRAMES = 75
# Half size (m) of the zoomed view around the robot.
ZOOM_HALF_SIZE = 0.4


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def quat_to_matrix(q):
    """[..., (qx, qy, qz, qw)] -> [..., 3, 3]."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def poses_to_rt(poses):
    """[N, 7] (x y z qx qy qz qw) -> (R [N,3,3], t [N,3])."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 7)
    return quat_to_matrix(poses[:, 3:]), poses[:, :3].copy()


def apply(transform, r, t):
    """world_from_x (R0, t0) composed with poses (R, t)."""
    r0, t0 = transform
    return np.einsum("ij,njk->nik", r0, r), t @ r0.T + t0


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


class Run:
    def __init__(self, path):
        nodes, opts = [], []
        self.final = None
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                kind = d["type"]
                if kind == "header":
                    self.t0 = d["t0"]
                elif kind == "ground_truth":
                    gt = np.array(d["poses"], dtype=float).reshape(-1, 8)
                elif kind == "node":
                    nodes.append(d)
                elif kind == "optimization":
                    d["after_nodes"] = len(nodes)
                    opts.append(d)
                elif kind == "final":
                    self.final = d
        if self.final is None:
            sys.exit("the recording has no 'final' record (run did not complete)")
        if [n["id"] for n in nodes] != list(range(len(nodes))):
            sys.exit("expected node ids 0..N-1 in order")
        self.n = len(nodes)
        self.times = np.array([n["t"] for n in nodes]) - self.t0
        w_map = poses_to_rt(self.final["world_from_map"])
        w_local = poses_to_rt(self.final["world_from_local"])
        self.w_map = (w_map[0][0], w_map[1][0])
        self.w_local = (w_local[0][0], w_local[1][0])

        self.live_r, self.live_t = apply(
            self.w_map, *poses_to_rt([n["global"] for n in nodes])
        )
        _, local_t = apply(self.w_local, *poses_to_rt([n["local"] for n in nodes]))
        self.local_t = local_t

        # Scan points in the node frame, concatenated.
        counts = [len(n["points"]) // 3 for n in nodes]
        self.point_node = np.repeat(np.arange(self.n), counts)
        self.points = np.concatenate(
            [np.array(n["points"], dtype=float).reshape(-1, 3) for n in nodes]
        )

        # Ground truth.
        self.gt_t = gt[:, 0] - self.t0
        self.gt_xyz = gt[:, 1:4]
        self.gt_at_nodes = np.stack(
            [np.interp(self.times, self.gt_t, self.gt_xyz[:, k]) for k in range(3)], 1
        )

        # Optimizations: the full pose snapshot, applied on top of the live
        # estimates.
        self.opts = []
        for o in opts:
            snap = np.array(o["nodes"], dtype=float).reshape(-1, 8)
            ids = snap[:, 0].astype(int)
            r, t = apply(self.w_map, *poses_to_rt(snap[:, 1:]))
            sub = np.array(o["submaps"], dtype=float).reshape(-1, 8)
            _, sub_t = apply(self.w_map, *poses_to_rt(sub[:, 1:]))
            cons = np.array(o["constraints"], dtype=int).reshape(-1, 3)
            self.opts.append(
                dict(
                    index=o["index"],
                    t=o["t"] - self.t0,
                    wall=o["wall"],
                    after_nodes=o["after_nodes"],
                    ids=ids,
                    r=r,
                    t_pose=t,
                    submap_ids=sub[:, 0].astype(int),
                    submap_t=sub_t,
                    constraints=cons,
                    counts=o["counts"],
                )
            )
        # Best pose of every node after k optimizations (k = 0..len(opts)).
        self.best = [(self.live_r.copy(), self.live_t.copy())]
        for o in self.opts:
            r, t = self.best[-1][0].copy(), self.best[-1][1].copy()
            r[o["ids"]] = o["r"]
            t[o["ids"]] = o["t_pose"]
            self.best.append((r, t))

        # Error series.
        self.err_live = np.linalg.norm(self.live_t - self.gt_at_nodes, axis=1)
        self.err_local = np.linalg.norm(self.local_t - self.gt_at_nodes, axis=1)

        # Ceiling: most populated 10 cm slice of the final map.
        world = self.world_points(*self.best[-1], self.n)
        hist = np.floor(world[:, 2] / 0.1).astype(int)
        values, counts = np.unique(hist, return_counts=True)
        self.ceiling = (values[np.argmax(counts)] + 0.5) * 0.1
        self.cut = self.ceiling - 0.3
        below = world[world[:, 2] < self.cut]
        everything = np.concatenate([below[:, :2], self.gt_xyz[:, :2]])
        self.xmin, self.ymin = everything.min(0) - 0.4
        self.xmax, self.ymax = everything.max(0) + 0.4
        self.zmin = min(below[:, 2].min(), self.gt_xyz[:, 2].min())
        self.err_max = max(self.err_live.max(), self.err_local.max())

    def world_points(self, r, t, num_nodes):
        """World coordinates of the points of nodes < num_nodes."""
        m = np.searchsorted(self.point_node, num_nodes)
        idx = self.point_node[:m]
        return np.einsum("nij,nj->ni", r[idx], self.points[:m]) + t[idx]


# ----------------------------------------------------------------------
# Frame plan
# ----------------------------------------------------------------------


def frame_plan(run):
    """List of (num_nodes, opt_k, phase, phase_fraction): `opt_k`
    optimizations applied; `phase` is None, "morph" (blend from opt_k - 1 to
    opt_k) or "hold" (showing the constraints of opt_k)."""
    plan = []
    k = 0
    for i in range(1, run.n + 1):
        plan.append((i, k, None, 0.0))
        while k < len(run.opts) and run.opts[k]["after_nodes"] <= i:
            k += 1
            prev = run.opts[k - 2]["counts"]["loop_closures"] if k >= 2 else 0
            new_loops = run.opts[k - 1]["counts"]["loop_closures"] - prev
            plan += [(i, k, "morph", (f + 1) / MORPH_FRAMES) for f in range(MORPH_FRAMES)]
            hold = HOLD_FRAMES if new_loops > 0 else HOLD_FRAMES_NO_LOOP
            plan += [(i, k, "hold", (f + 1) / hold) for f in range(hold)]
    plan += [(run.n, k, "end", (f + 1) / END_FRAMES) for f in range(END_FRAMES)]
    return plan


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def smoothstep(x):
    return x * x * (3 - 2 * x)


def rasterize(run, pts, cell=0.05):
    """RGBA image (rows = y) of points below the ceiling, coloured by mean
    height, opacity by density."""
    pts = pts[pts[:, 2] < run.cut]
    nx = int(np.ceil((run.xmax - run.xmin) / cell))
    ny = int(np.ceil((run.ymax - run.ymin) / cell))
    img = np.zeros((ny, nx, 4))
    if len(pts) == 0:
        return img
    ix = np.clip(((pts[:, 0] - run.xmin) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - run.ymin) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    count = np.bincount(flat, minlength=nx * ny)
    zsum = np.bincount(flat, weights=pts[:, 2], minlength=nx * ny)
    has = count > 0
    z = np.zeros(nx * ny)
    z[has] = zsum[has] / count[has]
    rgba = CMAP(np.clip((z - run.zmin) / (run.cut - run.zmin), 0, 1)).reshape(-1, 4)
    rgba[:, 3] = np.where(has, np.clip(0.35 + 0.65 * np.sqrt(count / 6.0), 0, 1), 0)
    return rgba.reshape(ny, nx, 4)


def style_axes(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


class Renderer:
    def __init__(self, run):
        self.run = run
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "text.color": FG,
                "axes.labelcolor": MUTED,
                "savefig.facecolor": BG,
            }
        )

    def poses(self, k, phase, frac):
        run = self.run
        if phase == "morph" and k >= 1:
            a = smoothstep(frac)
            (r0, t0), (r1, t1) = run.best[k - 1], run.best[k]
            return (1 - a) * r0 + a * r1, (1 - a) * t0 + a * t1
        return run.best[k]

    def zoom(self, fig, run, num, t, now):
        """Magnified view around the robot: the drift and the corrections
        are a few centimeters, invisible at room scale."""
        axz = fig.add_axes([0.865, 0.56, 0.125, 0.33])
        style_axes(axz)
        c = np.array([np.interp(now, run.gt_t, run.gt_xyz[:, k]) for k in range(2)])
        half = ZOOM_HALF_SIZE
        gt_mask = (run.gt_t <= now) & (run.gt_t >= now - 6.0)
        recent = run.times[:num] >= now - 6.0
        axz.plot(*run.gt_xyz[gt_mask, :2].T, color=GT_COLOR, lw=4, alpha=0.6)
        axz.plot(*run.local_t[:num][recent, :2].T, color=LOCAL_COLOR, lw=1.3, ls=(0, (4, 2)))
        axz.plot(*run.live_t[:num][recent, :2].T, ".", color=BEST_COLOR, ms=2.5)
        axz.plot(*t[:num][recent, :2].T, color=EST_COLOR, lw=1.8)
        axz.scatter(*t[num - 1, :2], s=50, c=EST_COLOR, edgecolors="white", lw=1.2, zorder=5)
        axz.set_xlim(c[0] - half, c[0] + half)
        axz.set_ylim(c[1] - half, c[1] + half)
        axz.set_aspect("equal")
        axz.set_xticks([])
        axz.set_yticks([])
        # 10 cm scale bar.
        x0, y0 = c[0] - half + 0.06 * half, c[1] - half + 0.08 * half
        axz.plot([x0, x0 + 0.1], [y0, y0], color=FG, lw=2)
        axz.text(x0 + 0.05, y0 + 0.03 * half, "10 cm", color=FG, fontsize=8, ha="center",
                 va="bottom")
        fig.text(0.865, 0.9, f"zoom: ±{100 * half:.0f} cm at robot", color=FG, fontsize=10.5)
        fig.text(0.865, 0.535, "orange dots: live estimates", color=MUTED, fontsize=8.5)

    def render(self, frame, path):
        run = self.run
        num, k, phase, frac = frame
        r, t = self.poses(k, phase, frac)
        now = run.times[num - 1]

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)

        # Title.
        fig.text(
            0.015,
            0.962,
            "Cartographer 3D (pure Rust port) on a simulated Unitree L1 LiDAR + IMU",
            fontsize=17,
            weight="bold",
            color=FG,
        )
        fig.text(
            0.015,
            0.928,
            "local SLAM builds submaps scan by scan and drifts; global optimization "
            "with loop closures pulls the trajectory and map back into place",
            fontsize=11,
            color=MUTED,
        )

        # ---------------- top-down ----------------
        ax = fig.add_axes([0.035, 0.135, 0.585, 0.75])
        style_axes(ax)
        pts = run.world_points(r, t, num)
        img = rasterize(run, pts)
        ax.imshow(
            img,
            origin="lower",
            extent=(run.xmin, run.xmax, run.ymin, run.ymax),
            interpolation="nearest",
            zorder=1,
        )
        # Current scan highlighted.
        lo = np.searchsorted(run.point_node, num - 1)
        hi = np.searchsorted(run.point_node, num)
        scan = pts[lo:hi]
        scan = scan[scan[:, 2] < run.cut]
        ax.scatter(scan[:, 0], scan[:, 1], s=3, c="white", alpha=0.9, lw=0, zorder=6)

        gt_mask = run.gt_t <= now
        ax.plot(*run.gt_xyz[gt_mask, :2].T, color=GT_COLOR, lw=3.2, alpha=0.75, zorder=3,
                label="ground truth")
        ax.plot(*run.local_t[:num, :2].T, color=LOCAL_COLOR, lw=1.1, ls=(0, (4, 2)),
                alpha=0.9, zorder=4, label="local SLAM only (no loop closure)")
        ax.plot(*t[:num, :2].T, color=EST_COLOR, lw=1.6, zorder=5,
                label="SLAM estimate (current)")

        # Constraints + submaps after an optimization.
        banner = None
        if phase in ("hold", "end", "morph") and k >= 1:
            o = run.opts[k - 1]
            prev_loops = run.opts[k - 2]["counts"]["loop_closures"] if k >= 2 else 0
            loops = o["counts"]["loop_closures"]
            if phase == "hold":
                fade = 1.0 - smoothstep(max(0.0, (frac - 0.6) / 0.4))
            elif phase == "morph":
                fade = smoothstep(frac) * 0.6
            else:
                fade = 0.0
            if fade > 0 and len(o["constraints"]):
                sub_index = {s: n for n, s in enumerate(o["submap_ids"])}
                cons = o["constraints"]
                node_xy = t[cons[:, 0], :2]
                sub_xy = np.array([o["submap_t"][sub_index[s], :2] for s in cons[:, 1]])
                is_loop = cons[:, 2] == 1
                for mask, color, alpha, lw in (
                    (~is_loop, INTER_COLOR, 0.25, 0.6),
                    (is_loop, LOOP_COLOR, 0.55, 0.8),
                ):
                    if mask.any():
                        segs = np.stack([node_xy[mask], sub_xy[mask]], 1)
                        ax.add_collection(
                            LineCollection(segs, colors=color, alpha=alpha * fade,
                                           lw=lw, zorder=7)
                        )
                ax.scatter(*o["submap_t"][:, :2].T, s=70, marker="s", c=SUBMAP_COLOR,
                           edgecolors="black", lw=0.8, alpha=fade, zorder=8)
            if phase in ("hold", "morph"):
                banner = (
                    f"global optimization #{o['index']}  —  {loops} loop closures"
                    + (f"  (+{loops - prev_loops} new)" if loops > prev_loops else "")
                )
        # Sensor pose.
        heading = r[num - 1][:2, 0]
        p = t[num - 1]
        ax.scatter([p[0]], [p[1]], s=90, c=EST_COLOR, edgecolors="white", lw=1.5, zorder=9)
        ax.arrow(p[0], p[1], 0.7 * heading[0], 0.7 * heading[1], width=0.06,
                 head_width=0.28, head_length=0.25, color="white", zorder=9, lw=0)

        ax.set_xlim(run.xmin, run.xmax)
        ax.set_ylim(run.ymin, run.ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
        ax.set_title(
            f"top-down: map points below the ceiling (z < {run.cut:.1f} m), "
            "coloured by height, at each node's current pose estimate",
            color=FG, fontsize=11, loc="left",
        )
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([], [], color=LOOP_COLOR, lw=1.5),
            Line2D([], [], marker="s", ls="", color=SUBMAP_COLOR, markeredgecolor="black"),
            Line2D([], [], marker="o", ls="", color="white", markersize=3),
        ]
        labels += ["loop-closure constraint (node → submap)", "submap origin", "current scan"]
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.03, 0.005),
                   fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=FG,
                   framealpha=0.85, ncol=3)
        # Height colour bar.
        cax = fig.add_axes([0.505, 0.035, 0.11, 0.012])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(run.zmin, run.cut), cmap=CMAP
            ),
            cax=cax,
            orientation="horizontal",
        )
        cb.outline.set_edgecolor(GRID)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        fig.text(0.505, 0.054, "map point height z (m)", color=MUTED, fontsize=8.5)

        banner_color = LOOP_COLOR
        if phase == "end":
            banner = (
                f"done — ATE RMSE {100 * run.final['ate_rmse_optimized_m']:.1f} cm optimized "
                f"vs {100 * run.final['ate_rmse_local_m']:.1f} cm local SLAM only"
            )
            banner_color = GT_COLOR
        if banner:
            alpha = 1.0 if phase in ("morph", "end") else 1.0 - smoothstep(max(0.0, (frac - 0.75) / 0.25))
            ax.text(
                0.5, 0.955, banner, transform=ax.transAxes, ha="center", va="top",
                fontsize=17, weight="bold", color="black", alpha=alpha, zorder=30,
                bbox=dict(boxstyle="round,pad=0.45", fc=banner_color, ec="none", alpha=0.92 * alpha),
            )

        # ---------------- 3D ----------------
        self.zoom(fig, run, num, t, now)

        ax3 = fig.add_axes([0.63, 0.5, 0.225, 0.42], projection="3d")
        ax3.patch.set_alpha(0.0)
        ax3.set_axis_off()
        # Floor outline for orientation.
        fx = [run.xmin, run.xmax, run.xmax, run.xmin, run.xmin]
        fy = [run.ymin, run.ymin, run.ymax, run.ymax, run.ymin]
        ax3.plot(fx, fy, [0] * 5, color=GRID, lw=1)
        sub = pts[pts[:, 2] < run.cut]
        step = max(1, len(sub) // 14000)
        sub = sub[::step]
        ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.6, c=sub[:, 2], cmap=CMAP,
                    vmin=run.zmin, vmax=run.cut, depthshade=False, lw=0)
        ax3.plot(*t[:num].T, color=EST_COLOR, lw=1.4)
        ax3.scatter(*t[num - 1], s=30, c="white")
        ax3.set_xlim(run.xmin, run.xmax)
        ax3.set_ylim(run.ymin, run.ymax)
        ax3.set_zlim(0, run.ceiling)
        ax3.set_box_aspect((run.xmax - run.xmin, run.ymax - run.ymin, run.ceiling * 1.4),
                           zoom=1.25)
        ax3.view_init(elev=32, azim=-60 + 0.35 * self.frame_number)
        fig.text(0.645, 0.9, "3D map (rotating, ceiling hidden)", color=FG, fontsize=10.5)

        # ---------------- error vs time ----------------
        axe = fig.add_axes([0.675, 0.285, 0.31, 0.215])
        style_axes(axe)
        t_end = run.times[-1] + 1
        best_err = np.linalg.norm(t[:num] - run.gt_at_nodes[:num], axis=1)
        axe.plot(run.times[:num], run.err_local[:num], color=LOCAL_COLOR, lw=1.2,
                 label="local SLAM only")
        axe.plot(run.times[:num], run.err_live[:num], color=BEST_COLOR, lw=0.8, alpha=0.75,
                 label="live estimate (when the scan arrived)")
        axe.plot(run.times[:num], best_err, color=EST_COLOR, lw=1.8,
                 label="SLAM trajectory as currently estimated")
        for o in run.opts[:k]:
            axe.axvline(o["t"], color=LOOP_COLOR if o["counts"]["loop_closures"] else MUTED,
                        lw=0.8, ls=":", alpha=0.8)
        axe.set_xlim(0, t_end)
        axe.set_ylim(0, run.err_max * 1.08)
        axe.set_xlabel("time (s)", fontsize=9)
        axe.set_ylabel("translation error (m)", fontsize=9)
        axe.grid(color=GRID, lw=0.5)
        axe.set_title("translation error vs ground truth   (dotted: global optimizations)",
                      color=FG, fontsize=10, loc="left")
        axe.legend(loc="upper left", fontsize=8, facecolor=BG, edgecolor=GRID,
                   labelcolor=FG, framealpha=0.85)

        # ---------------- stats ----------------
        ate_best = float(np.sqrt(np.mean(best_err ** 2)))
        ate_local = float(np.sqrt(np.mean(run.err_local[:num] ** 2)))
        o = run.opts[k - 1] if k >= 1 else None
        submaps = o["counts"]["submaps"] if o else 1
        loops = o["counts"]["loop_closures"] if o else 0
        inter = o["counts"]["inter_submap"] if o else 0
        rows = [
            ("sim time", f"{now:5.1f} s / {run.times[-1]:.1f} s"),
            ("nodes (scans)", f"{num}"),
            ("submaps", f"{submaps}"),
            ("global optimizations", f"{o['index'] if o else 0}"),
            ("inter-submap constraints", f"{inter}  ({loops} close loops)"),
            ("ATE RMSE so far, SLAM", f"{100 * ate_best:5.1f} cm"),
            ("ATE RMSE so far, local only", f"{100 * ate_local:5.1f} cm"),
        ]
        x0, y0 = 0.645, 0.185
        for n, (label, value) in enumerate(rows):
            y = y0 - n * 0.026
            fig.text(x0, y, label, color=MUTED, fontsize=11, family="DejaVu Sans Mono")
            color = EST_COLOR if "SLAM" in label else (LOCAL_COLOR if "local" in label else FG)
            fig.text(x0 + 0.17, y, value, color=color, fontsize=11, weight="bold",
                     family="DejaVu Sans Mono")

        fig.savefig(path, dpi=DPI, facecolor=BG)
        plt.close(fig)


_RENDERER = None


def _init(record):
    global _RENDERER
    _RENDERER = Renderer(Run(record))


def _render(job):
    n, frame, path = job
    _RENDERER.frame_number = n
    _RENDERER.render(frame, path)
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("record", help="JSON-lines file from unitree_l1_bag --record")
    parser.add_argument("-o", "--output", default="target/unitree_l1/slam_run.mp4")
    parser.add_argument("--keyframes", help="directory for a few keyframe PNGs")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--frames", help="only render these frame numbers (comma separated) "
                        "as PNGs into --keyframes, no video")
    args = parser.parse_args()

    run = Run(args.record)
    plan = frame_plan(run)
    print(f"{run.n} nodes, {len(run.opts)} optimizations, {len(plan)} frames "
          f"= {len(plan) / FPS:.1f} s at {FPS} fps")

    if args.frames:
        out = args.keyframes or "."
        os.makedirs(out, exist_ok=True)
        _init(args.record)
        for n in map(int, args.frames.split(",")):
            path = os.path.join(out, f"frame_{n:05d}.png")
            _render((n, plan[n], path))
            print(path)
        return

    tmp = tempfile.mkdtemp(prefix="slam_frames_")
    try:
        jobs = [(n, f, os.path.join(tmp, f"{n:05d}.png")) for n, f in enumerate(plan)]
        with mp.Pool(args.jobs, initializer=_init, initargs=(args.record,)) as pool:
            for done, _ in enumerate(pool.imap_unordered(_render, jobs, chunksize=4)):
                if done % 100 == 0:
                    print(f"  {done}/{len(jobs)} frames", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-preset", "slow",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output],
            check=True,
        )
        print(f"wrote {args.output}")
        if args.keyframes:
            os.makedirs(args.keyframes, exist_ok=True)
            picks = keyframes(run, plan)
            for name, n in picks.items():
                dst = os.path.join(args.keyframes, f"{name}.png")
                shutil.copy(jobs[n][2], dst)
                print(f"  keyframe {dst}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def keyframes(run, plan):
    """A few representative frame numbers."""
    picks = {}
    first_loop = next((k for k, o in enumerate(run.opts, 1) if o["counts"]["loop_closures"]), None)
    for n, (num, k, phase, frac) in enumerate(plan):
        if "early" not in picks and num >= run.n // 8:
            picks["early"] = n
        if first_loop and k == first_loop and phase == "morph" and "before_loop" not in picks:
            picks["before_loop"] = n - 1
        if first_loop and k == first_loop and phase == "hold" and frac >= 0.3 and "loop_closure" not in picks:
            picks["loop_closure"] = n
        if "mid" not in picks and num >= run.n * 3 // 5 and phase is None:
            picks["mid"] = n
    picks["final"] = len(plan) - 1
    return picks


if __name__ == "__main__":
    main()
