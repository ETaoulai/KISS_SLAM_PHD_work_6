#!/usr/bin/env python3
"""Final ablation of the locked method C (#248b) on all 84 sequences: table + figure, from the runs of kiss_runs/q248/run_abl.sh.

    python scripts/ablation_table_248.py [--out=/home/photogrammetry/kiss_runs/ablation_248] [--seeds=0,1,2,3] [--jobs=8] [--fig=docs/figures/paper]

Rows (run prefix <variant>248, current code, explicit flags in kiss_runs/q248/common.sh):
  cumulative  KISS-SLAM · without deskew · + image motion (deskew + ICP start) · + two starts · + range-image fallback ·
              + upright SURF / guided matching · + blend for the deskew (B) · + KISS fallback after 4 failures = C (locked)
  leave-one-out from C: image as ICP start, image deskew, near-floor filter, fixed sigma, two starts, range fallback, upright / guided,
              blend, KISS fallback (= B)
Per group (special sequences apart): geometric mean of APE / APE(KISS-SLAM) over the sequences the row has, failures (APE > 5 m or crash;
cars: crash only), sequences better / worse than the reference row by > 5 % (cumulative: previous row; leave-one-out: C), median RPE 1 m
(hand-held: NCD + Spires).  Cars: full SLAM (loop closures) for every row.  Scored with paper_tables.py (official protocols, cache).
"""
import math
import sys
from pathlib import Path

sys.argv = [a for a in sys.argv]          # paper_tables reads --out / --seeds / --jobs from argv
if not any(a.startswith("--out=") for a in sys.argv):
    sys.argv.append("--out=/home/photogrammetry/kiss_runs/ablation_248")
sys.path.insert(0, str(Path(__file__).parent))
import paper_tables as pt  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CUM = [("KISS-SLAM", None), ("KISS-SLAM without deskew", "nodeskew248"), ("+ image motion (deskew + ICP start)", "img248"),
       ("+ two starts", "two248"), ("+ range-image fallback", "range248"), ("+ upright SURF / guided matching", "upright248"),
       ("+ blend for the deskew (B)", "blend248"), ("+ KISS fallback after 4 failures = C", "C")]
LOO = [("− image as ICP start (deskew only)", "nostart248"), ("− image deskew (ICP start only)", "nodesk248"), ("− near-floor filter", "nofloor248"),
       ("− fixed σ (KISS adaptive)", "adsigma248"), ("− two starts", "notwo248"), ("− range-image fallback", "norange248"),
       ("− upright SURF / guided matching", "noupr248"), ("− blend for the deskew", "noblend248"), ("− KISS fallback (= B)", "blend248")]
FIG = Path(pt.OPT.get("fig", Path(__file__).resolve().parent.parent / "docs/figures/paper"))


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    S = pt.sequences()
    for s in S:
        arms = {"KISS-SLAM": s["arms"]["KISS-SLAM"], "C (ours)": s["arms"]["C (ours)"]}
        for _, p in CUM[1:-1] + LOO:
            arms[p] = [p]
        s["arms"], s["replay"] = arms, None
    cells, tasks = pt.collect(S)
    res = pt.evaluate_cached(tasks)
    pt.aggregate(S, cells, res)
    key = lambda p: "KISS-SLAM" if p is None else "C (ours)" if p == "C" else p
    groups = [g for g in pt.GROUPS]
    seqs = {g: [s["seq"] for s in S if s["group"] == g and not s["special"]] for g in groups}

    def stats(g, p, ref):
        k, r = key(p), key(ref) if ref is not None or p is None else None
        ratios, fails, better, worse, n, seeds = [], 0, 0, 0, 0, []
        for q in seqs[g]:
            c, kc = cells.get((q, k)), cells.get((q, "KISS-SLAM"))
            if not c or c["status"] == "none":
                continue
            n += 1
            seeds.append(c["n"])
            fails += bool(c["fail"])
            if c["status"] == "ok" and kc and kc["status"] == "ok" and c["ate"] > 0 and kc["ate"] > 0:
                ratios.append(c["ate"] / kc["ate"])
            if ref is not None:
                rc = cells.get((q, key(ref)))
                if rc and rc["status"] == "ok" and c["status"] == "ok":
                    better += c["ate"] < 0.95 * rc["ate"]
                    worse += c["ate"] > 1.05 * rc["ate"]
        return dict(geo=geo(ratios), n=n, of=len(seqs[g]), fails=fails, better=better, worse=worse,
                    seeds=f"{min(seeds)}-{max(seeds)}" if seeds and min(seeds) != max(seeds) else str(seeds[0]) if seeds else "-")

    def rpe(p):
        k = key(p)
        t = [cells[(q, k)]["rpe_t"] for g in ("NCD", "Oxford Spires") for q in seqs[g] if (q, k) in cells and np.isfinite(cells[(q, k)].get("rpe_t", np.nan))]
        r = [cells[(q, k)]["rpe_r"] for g in ("NCD", "Oxford Spires") for q in seqs[g] if (q, k) in cells and np.isfinite(cells[(q, k)].get("rpe_r", np.nan))]
        return (float(np.median(t)), float(np.median(r)), len(t)) if t else (np.nan, np.nan, 0)

    L = ["# Final ablation of the locked method C (#248b) — all 84 sequences", "",
         "Geometric mean of APE / APE(KISS-SLAM) per group (sequences the row has / the group's), failures (APE > 5 m or crash; cars: crash only),",
         "better / worse than the reference row by > 5 % (cumulative: previous row; leave-one-out: C), RPE 1 m median over NCD + Spires.",
         "Seeds per cell in brackets. Special sequences (stairs, dynamic_spinning) excluded. Cars: full SLAM.", ""]
    head = "| row | " + " | ".join(f"{g}" for g in groups) + " | RPE 1 m t / r (hand-held) | failures | better / worse |"
    sep = "|---" * (len(groups) + 4) + "|"
    out_rows = []
    for title, rows, mode in (("Cumulative", CUM, "cum"), ("Leave-one-out from C", [("C (locked)", "C")] + LOO, "loo")):
        L += [f"## {title}", "", head, sep]
        for i, (label, p) in enumerate(rows):
            ref = (rows[i - 1][1] if i > 0 else None) if mode == "cum" else ("C" if p != "C" else None)
            st = {g: stats(g, p, ref) for g in groups}
            t, r, nr = rpe(p)
            fails = sum(st[g]["fails"] for g in groups)
            bw = f"{sum(st[g]['better'] for g in groups)} / {sum(st[g]['worse'] for g in groups)}" if ref is not None or (mode == "cum" and i > 0) else "–"
            cell = lambda s: "–" if not s["n"] else f"{s['geo']:.2f} ({s['n']}/{s['of']}, ×{s['seeds']})"
            L.append(f"| {label} | " + " | ".join(cell(st[g]) for g in groups) + f" | {t:.2f} / {r:.3f} | {fails} | {bw} |")
            out_rows.append(dict(mode=mode, label=label, prefix=p, **{g: st[g]["geo"] for g in groups}))
        L.append("")
    out = pt.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_248.md").write_text("\n".join(L) + "\n")
    import csv
    with open(out / "ablation_248.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("\n".join(L))
    figure(out_rows, groups)


def figure(rows, groups):
    BLUE, INK, INK2, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    cum = [r for r in rows if r["mode"] == "cum"]
    loo = [r for r in rows if r["mode"] == "loo" and r["prefix"] != "C"]
    c = next(r for r in rows if r["mode"] == "loo" and r["prefix"] == "C")
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, len(groups), height_ratios=[1.0, 1.25], hspace=0.55, wspace=0.35)
    for j, g in enumerate(groups):                      # (a) cumulative: one small multiple per group, one hue
        ax = fig.add_subplot(gs[0, j])
        y = [r[g] for r in cum]
        x = np.arange(len(y))
        ok = np.isfinite(y)
        ax.plot(x[ok], np.array(y)[ok], color=BLUE, lw=1.5, marker="o", ms=3)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_title(g, fontsize=7, color=INK, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(["K", "−d", "+I", "+2", "+R", "+U", "+B", "C"], fontsize=5.5)
        ax.grid(axis="y", color=GRID, lw=0.4, which="both")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if np.isfinite(y[-1]):
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6, color=INK)
    fig.text(0.0, 0.965, "(a) cumulative: APE / KISS-SLAM (geo-mean, log)   K KISS-SLAM · −d no deskew · +I image motion · +2 two starts · "
             "+R range fallback · +U upright SURF / guided · +B blend · C locked", fontsize=6.5, color=INK2)
    ax = fig.add_subplot(gs[1, :])                      # (b) leave-one-out: % change of the geo-mean vs C, diverging, value in every cell
    M = np.array([[100 * (r[g] / c[g] - 1) if np.isfinite(r[g]) and np.isfinite(c[g]) else np.nan for g in groups] for r in loo])
    lim = max(10.0, float(np.nanmax(np.abs(M)))) if np.isfinite(M).any() else 10.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f1f0ed", "#e34948"])   # better (blue) - gray - worse (red)
    norm = matplotlib.colors.SymLogNorm(linthresh=10, vmin=-lim, vmax=lim)
    ax.imshow(np.nan_to_num(M, nan=0.0), cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "–" if not np.isfinite(v) else f"{v:+.0f} %", ha="center", va="center", fontsize=6.5,
                    color=INK if not np.isfinite(v) or abs(norm(v) - 0.5) < 0.3 else "white")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7)
    ax.set_yticks(range(len(loo)))
    ax.set_yticklabels([r["label"] for r in loo], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(groups)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(loo)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.set_title("(b) leave-one-out: change of the geo-mean APE when the component is removed from C (positive = worse without it)",
                 fontsize=6.5, color=INK2, loc="left")
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"ablation.{ext}", bbox_inches="tight")
    print("wrote", FIG / "ablation.pdf")


if __name__ == "__main__":
    main()
