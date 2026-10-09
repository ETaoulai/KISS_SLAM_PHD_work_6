#!/usr/bin/env python3
"""Figures of the paper (#248), from the runs and the tables of scripts/paper_tables.py.

    python scripts/paper_figures.py [--csv=/home/photogrammetry/kiss_runs/paper_248/paper_248.csv] [--out=docs/figures/paper] [--only=traj,datasets,ablation]

traj      top view of GT, KISS-SLAM and C (seed 0) on three sequences: two where KISS-SLAM fails, one ordinary one; SE(3)-aligned to the GT
          on the associated poses (evo, t_max_diff 0.01 - the official protocol), APE in the labels.
datasets  per dataset group, geometric mean of APE / APE(KISS-SLAM) over the sequences each method has (special sequences left out; cars:
          odometry only, so every method is compared without loop closures): C and the best other LiDAR-only method, log scale.
ablation  the final ablation (#248b): cumulative and leave-one-out rows, geo-mean APE / KISS-SLAM per group (needs the ablation table csv).
Colours: the validated categorical palette (dataviz skill, light mode) - slot 1 blue = C, slot 2 orange = KISS-SLAM, GT in neutral gray;
every series is labelled directly, so colour is never the only cue.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evo.core import sync  # noqa: E402
from evo.core.trajectory import PoseTrajectory3D  # noqa: E402
from evo.tools import file_interface  # noqa: E402

OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
CSV = Path(OPT.get("csv", "/home/photogrammetry/kiss_runs/paper_248/paper_248.csv"))
OUT = Path(OPT.get("out", Path(__file__).resolve().parent.parent / "docs/figures/paper"))
ONLY = set(OPT.get("only", "traj,datasets").split(","))
EVAL = Path("/home/photogrammetry/kiss_runs/official_eval")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (validated, light)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titlecolor": INK, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def aligned(gt_file, est_file):
    ref = file_interface.read_tum_trajectory_file(str(gt_file))
    est = file_interface.read_tum_trajectory_file(str(est_file))
    ref_a, est_a = sync.associate_trajectories(ref, est, max_diff=0.01)
    est_al = PoseTrajectory3D(poses_se3=list(est_a.poses_se3), timestamps=est_a.timestamps)
    est_al.align(ref_a, correct_scale=False)
    ape = float(np.sqrt(np.mean(np.sum((est_al.positions_xyz - ref_a.positions_xyz) ** 2, axis=1))))
    return ref_a.positions_xyz, est_al.positions_xyz, ape


def fig_traj():
    cases = [("keble_03", "Oxford Spires\nkeble-college-03 (Hesai QT64)"),
             ("underground_hard", "Newer College\nunderground_hard (Ouster 128)"),
             ("quad_easy", "Newer College\nquad_easy (Ouster 128)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (seq, title) in zip(axes, cases):
        d = EVAL / seq
        gt = d / "gt_lidar.txt"
        G, K, ak = aligned(gt, d / "kiss_s0_posetime_tum.txt")
        _, C, ac = aligned(gt, d / "bfc150_s0_posetime_tum.txt")
        ax.plot(G[:, 0], G[:, 1], color=MUTED, lw=2.4, alpha=0.55, label="ground truth", solid_capstyle="round")
        ax.plot(K[:, 0], K[:, 1], color=ORANGE, lw=1.0, label="KISS-SLAM")
        ax.plot(C[:, 0], C[:, 1], color=BLUE, lw=1.0, label="ours (C)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{title}\nAPE: KISS-SLAM {ak:.2f} m · ours {ac:.2f} m", loc="left", fontsize=7.5)
        ax.set_xlabel("x [m]")
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("y [m]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "trajectories")


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def fig_datasets():
    d = pd.read_csv(CSV)
    d = d[~d.special.astype(bool)]
    rows = []
    for g in ["NCD", "Oxford Spires", "Hilti 2021", "Hilti 2022", "NTU VIRAL", "Boreas", "MulRan"]:
        x = d[d.group == g]
        car = g in ("Boreas", "MulRan")
        if car:
            x = x[x.block == "odometry"]
        kiss = "KISS-SLAM odo." if car else "KISS-SLAM"
        ours = "C odo. (ours)" if car else "C (ours)"
        k = x[x.method == kiss].set_index("sequence").ate
        res = {}
        for m, y in x[x.status == "ok"].groupby("method"):
            if m in (kiss, "KISS no deskew") or "IMU" in m or "FAST-LIO" in m or "COIN-LIO" in m:
                continue
            y = y.set_index("sequence").ate
            common = [s for s in y.index if s in k.index and k[s] > 0 and y[s] > 0]
            if len(common) >= max(1, len(k) // 2):
                res[m] = (geo([y[s] / k[s] for s in common]), len(common))
        if ours not in res:
            continue
        others = {m: v for m, v in res.items() if m != ours}
        best = min(others.items(), key=lambda kv: kv[1][0]) if others else None
        n = int(x[x.method == ours].sequence.nunique())
        rows.append((f"{g} ({n})", res[ours], best))
    fig, ax = plt.subplots(figsize=(4.6, 0.38 * len(rows) + 0.9))
    for i, (label, (rc, nc), best) in enumerate(rows):
        y = len(rows) - 1 - i
        if best:
            m, (rb, nb) = best
            ax.plot([min(rc, rb), max(rc, rb)], [y, y], color=GRID, lw=2, zorder=1)
            ax.scatter([rb], [y], s=36, color="white", edgecolor=INK2, lw=1.0, zorder=3)
            ax.annotate(f'{m.replace(" odo.", "")}, {nb}/{nc}', (rb, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        ax.scatter([rc], [y], s=40, color=BLUE, zorder=4)
        ax.annotate(f"{rc:.2f}", (rc, y), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    ax.axvline(1.0, color=ORANGE, lw=1.0, ls="--", zorder=0)
    ax.text(1.0, len(rows) - 0.4, "KISS-SLAM = 1", color=INK2, fontsize=6.5, ha="right", va="bottom")
    ax.set_xscale("log")
    ticks = [t for t in (0.1, 0.2, 0.5, 1.0, 2.0) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1] * 1.01] or [0.2, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("geometric mean of APE / APE(KISS-SLAM)  (log scale, lower is better)")
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.5, which="both")
    ax.scatter([], [], s=40, color=BLUE, label="ours (C, 4 seeds)")
    ax.scatter([], [], s=36, color="white", edgecolor=INK2, label="best other LiDAR-only method")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "datasets_vs_kiss")
    return rows


if __name__ == "__main__":
    if "traj" in ONLY:
        fig_traj()
    if "datasets" in ONLY:
        for r in fig_datasets():
            print(r)
