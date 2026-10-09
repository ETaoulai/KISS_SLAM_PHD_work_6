#!/usr/bin/env python3
"""The paper's result tables (#248): all 84 sequences, every method, each dataset's official protocol, from the runs on disk.

    python scripts/paper_tables.py --out=/home/photogrammetry/kiss_runs/paper_248 [--seeds=0,1,2,3] [--jobs=8] [--no-cache]

Writes <out>/<name of out>.md (tables), .csv (one row per sequence x method) and _coverage.txt (also printed): runs found per
cell, crashes, runs covering < 95 % of the GT time span, cells of ours with fewer than 4 seeds.  A json per run under
<out>/cache makes a rerun fast (key: trajectory / config / GT files with size and mtime, the protocol, CACHE_VERSION).
Nothing is written inside the run folders: evaluate_official writes the estimates with their pose times to
~/kiss_runs/official_eval/<seq>/ (as results_table.py does); the odometry replays and the IMU-frame copies used for the
dense Hilti references go to <out>/official_eval_odometry/, <out>/imu_frame/ and <out>/official_eval_imu/.

Decisions M.T. 9/10 (#248).  84 sequences: NCD 12 (10 + stairs, dynamic_spinning shown apart), Oxford Spires 13, Hilti 2021 12 and
2022 16, NTU VIRAL 18, Boreas 9 (8 full drives + the 3000-scan cut of 2021-01-26-11-22), MulRan 4.  Ours = method C, seeds
s0-s3 only (mean ± σ); the other methods their run(s) in s0-s3.  Arms of C: bfc150 (the 42-table: NCD, Spires 6, NTU 18,
Boreas cut), ral228 (Spires 7 new), ral239 (Hilti 28), ral225 (Boreas 8, one run), ral223 (MulRan).  Hilti uses the #239-#241
arms only (kiss239, ral239, genz240, mad240, cticp240, trajlo240) for all 28 - not the older 42-table arms of the first 6.
Metrics (scripts reused, not re-implemented):
  APE      evaluate_official.py (evo APE --align, t_max_diff 0.01) on NCD / Spires / Boreas / MulRan; evaluate_hilti.py (2021);
           evaluate_hilti2022.py (2022, + challenge score 0-100); evaluate_ntu.py (prism ATE).
  RTE %    KITTI 100-800 m (kiss_icp.metrics.sequence_error, as evaluate_official.py) where the GT is a dense trajectory with
           orientation: NCD, Spires, Boreas, MulRan and the dense Hilti 2022 references (exp14 / 16 / 18 *_imu.txt, 10 Hz) - there
           scored by evaluate_official.py on the IMU-frame trajectory (pose x (T_imu_lidar)^-1, calibration of evaluate_hilti2022).
           n/a below 100 m of path, on the sparse Hilti control points, on the NTU prism positions, and on the two dense Hilti 2021
           references (10 ms-rounded duplicated stamps, UZH with 10 m jumps inside one stamp: APE only, the challenge's metric).
  RPE 1 m  translation cm / rotation deg, evaluate_official.py, hand-held only (NCD, Spires, dense Hilti 2022); never on the cars.
Cars (Boreas, MulRan): SLAM (with loop closures) and odometry blocks; the odometry of KISS-SLAM / C are the existing replays
without closures (replay_backend.py none: ~/kiss_runs/backend_replay_225, _223), the other methods have no closures.
Marks: † APE > 5 m (hand-held / drone groups; not on the 8 km car drives, as docs/results.md §4) · ✗ crash = the run exists
(log / folder) but wrote no trajectory (Traceback, segfault, out of memory) - a failure · ∞ = NTU run that stopped early
(official rule) · – = never run there · n/a = metric undefined.
"""
import csv
import hashlib
import json
import math
import re
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
if any(not a.startswith("--") for a in sys.argv[1:]):
    raise SystemExit(__doc__)       # results_table.py reads the first positional argument as its data root
from evaluate_gt import find_tum  # noqa: E402
from evaluate_hilti import T_imu_os_sensor, evaluate_hilti  # noqa: E402
from evaluate_hilti2022 import T_imu_lidar, evaluate_hilti2022  # noqa: E402
from evaluate_ncd import load_gt  # noqa: E402
from evaluate_ntu import evaluate_ntu, load_gt as ntu_gt  # noqa: E402
from evaluate_official import evaluate_official, pose_times, write_tum  # noqa: E402
from hilti_table import gt_2021, gt_2022  # noqa: E402
import results_table  # noqa: E402   (SEQUENCES: GT paths and run folders of NCD / Spires)

OPT = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:])
OUT = Path(OPT.get("out", "/home/photogrammetry/kiss_runs/paper_248"))
SEEDS = None if OPT.get("seeds", "0,1,2,3") == "all" else {int(s) for s in OPT.get("seeds", "0,1,2,3").split(",")}
JOBS = int(OPT.get("jobs", 8))
CACHE_VERSION = 1
EXTRA = Path("/home/photogrammetry/kiss_runs")
NTFS_RUNS = Path("/media/photogrammetry/A26C3DDF6C3DAF431/data/runs")
ROOTS = [NTFS_RUNS, EXTRA, Path("/home/photogrammetry/kiss_runs_ssd")]       # later roots win on a name clash (as results_table)
SSD = Path("/media/photogrammetry/Extreme SSD")
FAIL = 5.0

LIDAR = ["KISS-SLAM", "KISS no deskew", "GenZ-ICP", "MAD-ICP", "DLO", "CT-ICP", "Traj-LO", "C (ours)"]
IMU = ["FAST-LIO2 (IMU)", "COIN-LIO (IMU)"]
ODO = ["KISS-SLAM odo.", "C odo. (ours)", "CT-ICP", "MAD-ICP", "GenZ-ICP"]          # car odometry block
ARMS42 = {"KISS-SLAM": ["kiss"], "KISS no deskew": ["kissnodeskew"], "GenZ-ICP": ["genz"], "MAD-ICP": ["mad"], "DLO": ["dlo"],
          "CT-ICP": ["cticp"], "Traj-LO": ["trajlo"], "C (ours)": ["bfc150"], "FAST-LIO2 (IMU)": ["fastlio"], "COIN-LIO (IMU)": ["coinlio"]}
ARMS_SPIRES_NEW = {"KISS-SLAM": ["kiss228"], "C (ours)": ["ral228"]}
ARMS_HILTI = {"KISS-SLAM": ["kiss239"], "GenZ-ICP": ["genz240"], "MAD-ICP": ["mad240"], "CT-ICP": ["cticp240"], "Traj-LO": ["trajlo240"],
              "C (ours)": ["ral239"]}
ARMS_BOREAS = {"KISS-SLAM": ["kiss216"], "C (ours)": ["ral225"], "CT-ICP": ["cticp225", "cticp220"], "MAD-ICP": ["mad225", "mad220"],
               "GenZ-ICP": ["genz225", "genz220"]}
ARMS_MULRAN = {"KISS-SLAM": ["kiss208", "kiss210", "kiss215"], "C (ours)": ["ral223"], "CT-ICP": ["cticp224"], "MAD-ICP": ["mad224"],
               "GenZ-ICP": ["genz224"]}
ODO_OF = {"KISS-SLAM odo.": "KISS-SLAM", "C odo. (ours)": "C (ours)"}
GROUPS = ["NCD", "Oxford Spires", "Hilti 2021", "Hilti 2022", "NTU VIRAL", "Boreas", "MulRan"]
CARS = {"Boreas", "MulRan"}
SPECIAL = {"stairs", "dynamic_spinning"}


def sequences():
    """The 84 sequences: dict(group, seq, kind, gt, frame, rel, arms, special, dense, replay)."""
    S = []
    spires_new = {"keble_02", "keble_04", "keble_05", "observatory_02", "blenheim_01", "blenheim_05", "church_05"}
    for dataset, seq, _, gt, frame, folder in results_table.SEQUENCES:
        if dataset.startswith("Newer College") or dataset == "Oxford Spires":
            rel = folder.relative_to(results_table.RUNS)
            S.append(dict(group="NCD" if dataset.startswith("Newer") else "Oxford Spires", seq=seq, kind="official", gt=gt, frame=frame,
                          rel=rel, arms=ARMS_SPIRES_NEW if rel.name in spires_new else ARMS42, special=seq in SPECIAL))
    for line in open(EXTRA / "all_seqs.tsv"):
        q, _, rel = line.split("\t")[:3]
        if rel.startswith("hilti_2021/"):
            S.append(dict(group="Hilti 2021", seq=q, kind="hilti21", gt=gt_2021(q), rel=Path(rel), arms=ARMS_HILTI))
    for line in open(EXTRA / "hilti_new.tsv"):
        q, _, rel = line.split("\t")[:3]
        if rel.startswith("hilti_2021/"):
            S.append(dict(group="Hilti 2021", seq=q, kind="hilti21", gt=gt_2021(q), rel=Path(rel), arms=ARMS_HILTI))
        else:
            sparse, dense = gt_2022(q)
            S.append(dict(group="Hilti 2022", seq=q, kind="hilti22", gt=sparse, dense=dense, rel=Path(rel), arms=ARMS_HILTI))
    # Hilti 2021: the two dense *_imu.txt references (LAB_Survey_2, UZH_Tracking_Area_Run_2) stay APE-only (the challenge's metric):
    # stamps rounded to 10 ms with 2-10 rows per stamp, and in UZH poses of one stamp up to 10 m apart (GT path 138 m at 100 Hz,
    # 77 m at 10 Hz) - an RTE / RPE against them measures the reference's jitter (#248).  RTE / RPE only on the clean 10 Hz 2022 ones.
    for q in ("eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03", "rtp_01", "rtp_02", "rtp_03",
              "tnp_01", "tnp_02", "tnp_03", "spms_01", "spms_02", "spms_03"):
        S.append(dict(group="NTU VIRAL", seq=q, kind="ntu", rel=Path("ntu_viral") / q, arms=ARMS42))
    for q in ("2020-12-01-13-26", "2021-01-15-12-17", "2021-01-19-15-08", "2021-04-08-12-44", "2021-09-07-09-35", "2021-09-14-20-00",
              "2021-10-15-12-35", "2021-11-16-14-10"):
        S.append(dict(group="Boreas", seq=q, kind="official", frame="none", gt=SSD / "boreas" / f"gt_boreas-{q}_lidar_tum.txt",
                      rel=Path("boreas") / q, arms=ARMS_BOREAS, replay=EXTRA / "backend_replay_225" / "none" / q))
    S.append(dict(group="Boreas", seq="2021-01-26-11-22 (first 3000 scans)", kind="official", frame="none",
                  gt=SSD / "boreas" / "gt_boreas-2021-01-26-11-22_lidar_tum.txt", rel=Path("boreas/2021-01-26-11-22_3000"),
                  arms={k: ARMS42[k] for k in ("KISS-SLAM", "KISS no deskew", "C (ours)")}))
    for q in ("KAIST01", "DCC01", "Riverside01", "Sejong01"):
        S.append(dict(group="MulRan", seq=q, kind="official", frame="none", gt=SSD / "mulran" / q / "gt_lidar_tum.txt",
                      rel=Path("mulran") / q, arms=ARMS_MULRAN, replay=EXTRA / "backend_replay_223" / "none" / q))
    for s in S:
        s.setdefault("special", False), s.setdefault("dense", None), s.setdefault("replay", None)
    return S


# ---------------------------------------------------------------- runs on disk
_listing = {}


def listing(d):
    if d not in _listing:
        names = set()
        if d.is_dir():
            for p in d.iterdir():
                names.add(p.name[:-4] if p.name.endswith(".log") else p.name)
        _listing[d] = names
    return _listing[d]


def seed_ok(name):
    m = re.search(r"_s(\d+)$", name)
    return SEEDS is None or not m or int(m[1]) in SEEDS


def has_poses(run):
    return any(run.glob("*/*_poses_tum.txt"))


def find_runs(rel, prefixes):
    """{run name: (path, 'ok' | 'crash', note)} over the run roots; a finished run beats a crashed one of the same name."""
    found = {}
    for root in ROOTS:
        d = root / rel
        for n in sorted(listing(d)):
            if n.split("_")[0] not in prefixes or not re.search(r"_(s\d+|[a-z])$", n) or not seed_ok(n):
                continue
            p, log = d / n, d / f"{n}.log"
            txt = log.read_text(errors="replace") if log.exists() else ""
            note = "; ".join(w for w, k in (("Traceback in log", "Traceback"), ("killed", "Killed"), ("segfault", "Segmentation fault"),
                                            ("signal 11", "signal 11")) if k in txt)
            if p.is_dir() and has_poses(p):
                found[n] = (p, "ok", note)
            elif n not in found or found[n][1] != "ok":
                found[n] = (p, "crash", note or ("no log" if not log.exists() else "no trajectory"))
    return found


# ---------------------------------------------------------------- evaluation (one run)
@lru_cache(maxsize=None)
def gt_official(path, frame):
    t, T, _ = load_gt(path, frame)
    return t, T


def gt_span(task):
    if task["kind"] == "ntu":
        t = ntu_gt(task["seq"])[0]
    else:
        t = np.loadtxt(task["gt"], usecols=0, comments="#", ndmin=1)
    return float(t.min()), float(t.max())


def cover_of(run, span):
    t = pose_times(run, "none")[0]
    return max(0.0, min(t[-1], span[1]) - max(t[0], span[0])) / max(span[1] - span[0], 1e-9)


def imu_frame_copy(run, T_imu_sensor, seq):
    """The run's trajectory as the IMU's (pose x (T_imu_sensor)^-1), as an exact-time run folder under OUT (dense Hilti GT)."""
    t, T, _ = pose_times(run, "none")
    d = OUT / "imu_frame" / seq / run.name / "imu"
    d.mkdir(parents=True, exist_ok=True)
    T_imu = T @ np.linalg.inv(T_imu_sensor)
    write_tum(d / f"{run.name}_poses_posetime_tum.txt", t, T_imu)
    write_tum(d / f"{run.name}_poses_tum.txt", t, T_imu)
    return d.parent


def official(gt, frame, run, out_dir):
    gt_t, gt_T = gt_official(str(gt), frame)
    v = evaluate_official(gt_t, gt_T, run, frame, out_dir)
    return {k: float(v[k]) for k in ("ate", "rte", "rre", "rpe_t", "rpe_r", "cover", "path", "gt_path", "matched", "gt_interp")}


def evaluate(task):
    run, kind, seq = Path(task["run"]), task["kind"], task["seq"]
    if kind == "official":
        out_dir = (OUT / "official_eval_odometry" / seq) if task.get("odometry") else EXTRA / "official_eval" / run.parent.name
        return official(task["gt"], task["frame"], run, out_dir)
    if kind == "ntu":
        h = evaluate_ntu(seq, run)
        return dict(ate=float(h["ate"]), ate_raw=float(h["ate_raw"]), completeness=float(h["completeness"]),
                    cover=cover_of(run, gt_span(task)))
    if kind == "hilti21":
        h = evaluate_hilti(task["gt"], run)
        v = dict(ate=float(h["rmse"]), matched=h["n_matched"] / h["n_ref"], cover=cover_of(run, gt_span(task)))
        T_s = T_imu_os_sensor()
    else:
        h = evaluate_hilti2022(task["gt"], run)
        v = dict(ate=float(h["rmse"]), score=float(h.get("score", float("nan"))), matched=float(h["completeness"]),
                 cover=cover_of(run, gt_span(task)))
        T_s = T_imu_lidar()
        if task["dense"]:
            v["dense_ape"] = float(evaluate_hilti2022(task["dense"], run)["rmse"])
    if task["dense"]:                  # RTE / RPE against the dense IMU trajectory, in the IMU frame
        d = official(task["dense"], "none", imu_frame_copy(run, T_s, seq), OUT / "official_eval_imu" / seq)
        v.update({k: d[k] for k in ("rte", "rre", "rpe_t", "rpe_r", "path", "gt_path", "gt_interp")}, dense_cover=d["cover"])
    return v


def cache_key(task):
    run = Path(task["run"])
    stat = lambda f: f"{Path(f).resolve()}:{Path(f).stat().st_size}:{Path(f).stat().st_mtime_ns}" if f and Path(f).exists() else f"{f}:-"
    files = [f for pat in ("*_poses_tum.txt", "*_poses_posetime_tum.txt", "config.yml", "slam_config.yaml", "two_start.csv")
             for f in sorted(run.glob(f"**/{pat}"))]
    parts = [task["kind"], task.get("frame", ""), str(task.get("odometry", False)), str(CACHE_VERSION), stat(task.get("gt")),
             stat(task.get("dense")), task["seq"]] + [stat(f) for f in files]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def evaluate_cached(tasks):
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    res, todo = {}, []
    for k, task in tasks.items():
        f = cache / f"{cache_key(task)}.json"
        if f.exists() and "no-cache" not in OPT:
            res[k] = json.loads(f.read_text())
        else:
            todo.append((k, task, f))
    print(f"{len(tasks)} runs: {len(tasks) - len(todo)} cached, {len(todo)} to score ({JOBS} workers)", flush=True)
    with ProcessPoolExecutor(JOBS) as ex:
        futs = {ex.submit(evaluate, task): (k, f) for k, task, f in todo}
        for i, fu in enumerate(futs):
            k, f = futs[fu]
            try:
                v = fu.result()
            except Exception as e:                              # a run the protocol cannot score: reported, not hidden
                v = dict(error=f"{type(e).__name__}: {e}")
                print(f"  ERROR {k}: {v['error']}", flush=True)
            else:
                f.write_text(json.dumps(v))
            res[k] = v
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} / {len(todo)}", flush=True)
    return res


# ---------------------------------------------------------------- cells
def collect(S):
    """cells[(seq, method)] = dict(runs=[(name, path)], crash=[(name, note)], warn=[...]); and the evaluation tasks."""
    cells, tasks = {}, {}
    for s in S:
        for m, prefixes in s["arms"].items():
            found = find_runs(s["rel"], set(prefixes))
            c = cells[(s["seq"], m)] = dict(runs=[], crash=[], warn=[], ours=m == "C (ours)")
            for n, (p, status, note) in sorted(found.items()):
                if status == "ok":
                    c["runs"].append(n)
                    if note:
                        c["warn"].append(f"{n}: trajectory present but log has {note}")
                    tasks[(s["seq"], m, n)] = dict(run=str(p), seq=s["seq"], kind=s["kind"], gt=s.get("gt") and str(s["gt"]),
                                                   frame=s.get("frame", "none"), dense=s["dense"] and str(s["dense"]))
                else:
                    c["crash"].append(f"{n} ({note})")
        if s["replay"]:                                         # car odometry: the replay without closures of each SLAM run
            for mo, m in ODO_OF.items():
                c = cells[(s["seq"], mo)] = dict(runs=[], crash=[], warn=[], ours=mo == "C odo. (ours)")
                for n in cells[(s["seq"], m)]["runs"]:
                    p = s["replay"] / n
                    if p.is_dir() and has_poses(p):
                        c["runs"].append(n)
                        tasks[(s["seq"], mo, n)] = dict(run=str(p), seq=s["seq"], kind="official", gt=str(s["gt"]), frame="none",
                                                        dense=None, odometry=True)
                    else:
                        c["warn"].append(f"{n}: no odometry replay in {s['replay']}")
            for m in ("CT-ICP", "MAD-ICP", "GenZ-ICP"):          # odometry-only methods belong to the odometry block
                if (s["seq"], m) in cells:
                    cells[(s["seq"], m)]["odo"] = True
    return cells, tasks


def aggregate(S, cells, res):
    for s in S:
        for (q, m), c in cells.items():
            if q != s["seq"]:
                continue
            vals = [res[(q, m, n)] for n in c["runs"]]
            c["errors"] = [f"{n}: {v['error']}" for n, v in zip(c["runs"], vals) if "error" in v]
            vals = [v for v in vals if "error" not in v]
            c["n"] = len(vals)
            c["covers"] = [v.get("cover", np.nan) for v in vals]
            c["cover_low"] = [f"{n}: covers {v['cover']:.3f}" for n, v in zip(c["runs"], vals) if v.get("cover", 1) < 0.95]
            for k in ("ate", "rte", "rre", "rpe_t", "rpe_r", "score", "dense_ape", "path", "gt_path"):
                x = np.array([v.get(k, np.nan) for v in vals], float)
                fin = x[np.isfinite(x)]
                c[k] = float(fin.mean()) if len(fin) else (float("inf") if np.isinf(x).any() else float("nan"))
                c[k + "_sd"] = float(fin.std(ddof=1)) if len(fin) > 1 else float("nan")
            c["n_inf"] = int(sum(np.isinf(v.get("ate", np.nan)) for v in vals))
            c["status"] = ("crash" if not vals and c["crash"] else "inf" if vals and c["n_inf"] == len(vals) else "ok" if vals else "none")
            c["fail"] = c["status"] in ("crash", "inf") or (s["group"] not in CARS and c["status"] == "ok" and c["ate"] > FAIL)


# ---------------------------------------------------------------- formatting
def num(x, kind):
    if kind == "ape":
        return f"{x:.3f}" if x < 10 else f"{x:.1f}"
    if kind in ("rte", "rpe_t"):
        return f"{x:.2f}"
    if kind == "rpe_r":
        return f"{x:.3f}"
    return f"{x:.1f}"


def val(c, k, sd=True):
    x = c[k]
    if not np.isfinite(x):
        return None
    kind = "ape" if k in ("ate", "dense_ape") else k
    s = num(x, kind)
    if sd and c["ours"] and c["n"] > 1 and np.isfinite(c[k + "_sd"]):        # ± (σ over the seeds) for ours only, decimals of the mean
        s += " ± " + (f"{c[k + '_sd']:.1f}" if kind == "ape" and x >= 10 else num(c[k + "_sd"], kind))
    return s


def best_of(cells, seq, methods, k, lidar_only):
    xs = {m: cells[(seq, m)][k] for m in methods if (seq, m) in cells and m in lidar_only and cells[(seq, m)]["status"] == "ok"
          and np.isfinite(cells[(seq, m)][k])}
    return min(xs.values()) if len(xs) > 1 else None


def cell_text(s, c, metrics, best):
    if c is None:
        return "–"
    if c["status"] == "crash":
        return "✗ crash"
    if c["status"] == "none":
        return "–"
    if c["status"] == "inf":
        return "∞ (stopped early)"
    parts = []
    for k in metrics:
        if k == "rte" and not s["rte_defined"]:
            parts.append("n/a")
            continue
        t = val(c, k)
        if t is None:
            parts.append("n/a")
            continue
        if k == "ate" and s["group"] not in CARS and c[k] > FAIL:
            t += " †"
        if best.get(k) is not None and abs(c[k] - best[k]) < 1e-12:
            t = f"**{t}**"
        if k == "ate" and s["kind"] == "hilti22" and np.isfinite(c.get("score", np.nan)):
            t += f" ({c['score']:.1f})"
        parts.append(t)
    note = []
    if c["crash"]:
        note.append(f"{len(c['crash'])} ✗")
    if c["n_inf"]:
        note.append(f"{c['n_inf']} ∞")
    return " · ".join(parts) + (f" [{', '.join(note)}]" if note else "")


def main():
    if not NTFS_RUNS.is_dir():
        print(f"WARNING: {NTFS_RUNS} not mounted - KISS-SLAM / no-deskew runs of NCD / Spires live there", file=sys.stderr)
    S = sequences()
    assert len(S) == 84, len(S)
    for s in S:
        s["rte_defined"] = s["kind"] == "official" or s["dense"] is not None
    cells, tasks = collect(S)
    res = evaluate_cached(tasks)
    aggregate(S, cells, res)
    name = OUT.name
    L = [f"# Paper tables — 84 sequences, official protocols (#248, generated by scripts/paper_tables.py)", "",
         "Ours = method C (locked RA-L method), mean ± σ over seeds s0–s3; other methods: their runs (one each unless noted in the coverage "
         "report). APE = each dataset's official protocol (m): evo APE `--align --t_max_diff 0.01` on NCD / Oxford Spires / Boreas / MulRan "
         "(LiDAR frame, every pose at the instant it stands for), Hilti 2021 `evaluate_hilti.py`, Hilti 2022 `evaluate_hilti2022.py` "
         "(challenge score /100 in brackets), NTU VIRAL `evaluate_ntu.py` (prism ATE). RTE % = KITTI 100–800 m (`kiss_icp.metrics.sequence_error`) "
         "where the GT is a dense trajectory with orientation (dense Hilti references: on the IMU-frame trajectory); n/a on sparse control points / "
         "prism positions and below 100 m of path. Cell = APE · RTE. **Bold** = best LiDAR-only (cars: within the SLAM and within the odometry block). "
         "† APE > 5 m (not on the car drives) · ✗ crash = run exists but wrote no trajectory (failure) · ∞ = NTU run that stopped early "
         "(official rule) · – = not run · [k ✗] = k of the cell's runs crashed.", ""]
    rows = []
    for g in GROUPS:
        gs = [s for s in S if s["group"] == g]
        L += [f"## {g} ({len(gs)} sequences)", ""]
        if g in CARS:
            blocks = [("SLAM", [m for m in LIDAR if any((s["seq"], m) in cells and not cells[(s["seq"], m)].get("odo") for s in gs)]),
                      ("odometry", [m for m in ODO if any((s["seq"], m) in cells and (cells[(s["seq"], m)].get("odo") or m in ODO_OF)
                                                          for s in gs)])]
        else:
            ms = [m for m in LIDAR + IMU if any((s["seq"], m) in cells for s in gs)]
            blocks = [("", ms)]
        metrics = ["ate"] if g in ("NTU VIRAL", "Hilti 2021") else ["ate", "rte"]
        head = [f"{m} ({b})" if b else m for b, ms in blocks for m in ms]
        L += ["| sequence | " + " | ".join(head) + " |", "|---" * (len(head) + 1) + "|"]
        for s in gs:
            cs = []
            for b, ms in blocks:
                lo = [m for m in ms if m not in IMU]
                best = {k: best_of(cells, s["seq"], ms, k, lo) for k in metrics}
                for m in ms:
                    c = cells.get((s["seq"], m))
                    cs.append(cell_text(s, c, metrics, best))
                    if c is not None:
                        rows.append(dict(group=g, sequence=s["seq"], special=s["special"], block=b or "-", method=m, runs=c.get("n", 0),
                                         crashed=len(c["crash"]), status=c["status"], fail=c["fail"],
                                         **{k: c[k] for k in ("ate", "ate_sd", "rte", "rte_sd", "rpe_t", "rpe_t_sd", "rpe_r", "rpe_r_sd",
                                                              "score", "dense_ape", "path", "gt_path")}))
            gp = next((cells[(s["seq"], m)]["gt_path"] for b, ms in blocks for m in ms
                       if (s["seq"], m) in cells and np.isfinite(cells[(s["seq"], m)].get("gt_path", np.nan))), np.nan)
            label = s["seq"] + (f" ({gp:.0f} m)" if np.isfinite(gp) else "") + (" *special*" if s["special"] else "")
            if g == "Hilti 2022" and s["dense"]:
                label += " (dense)"
            L.append(f"| {label} | " + " | ".join(cs) + " |")
        if g == "NCD":
            L += ["", "*special* = stairs / dynamic_spinning: shown apart, left out of the summaries (special conditions)."]
        if g == "Hilti 2021":
            L += ["", "APE only (control points / the challenge's dense IMU references of LAB_Survey_2 and UZH_Tracking_Area_Run_2, whose "
                  "10 ms-rounded, duplicated stamps — UZH: poses of one stamp up to 10 m apart — make an RTE / RPE meaningless). Arms of #239–#241 "
                  "for all 12 (DLO / FAST-LIO2 / KISS no deskew of the 42-table exist only on the first 6 and are not used)."]
        if g == "Hilti 2022":
            L += ["", "APE against the sparse control points (official), score in brackets; RTE on the dense IMU trajectory of exp14 / 16 / 18 "
                  "(\"(dense)\"), n/a elsewhere."]
        if g in CARS:
            L += ["", "SLAM = with loop closures; odometry = KISS-SLAM / C without closures (existing `replay_backend.py none` replays), the other "
                  "methods (no closures). APE of an 8 km drive is not a failure test: no † here."
                  + (" The 3000-scan cut (§1 of docs/results.md) is scored on its own span (22 % of the drive's GT) and has no odometry replay (–)."
                     if g == "Boreas" else "")]
        L.append("")

    # ---- per-dataset summaries
    L += ["## Per-dataset summary (special sequences excluded)", "",
          "Geo-mean = geometric mean over the sequences where both have an APE (crashes / ∞ left out) of APE / KISS-SLAM "
          "(cars: SLAM block / KISS-SLAM, odometry block / KISS-SLAM odometry); same for RTE where defined. Wins = best LiDAR-only APE on the "
          "sequence (within the block). Failures = APE > 5 m (not cars) + crashes + ∞.", ""]
    L += ["| dataset | method | sequences run | geo-mean APE / KISS | geo-mean RTE / KISS | LiDAR-only wins | failures (crashes) | mean score |",
          "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        gs = [s for s in S if s["group"] == g and not s["special"]]
        blocks = ([("SLAM", [m for m in LIDAR], "KISS-SLAM"), ("odometry", ODO, "KISS-SLAM odo.")] if g in CARS
                  else [("", LIDAR + IMU, "KISS-SLAM")])
        for b, ms, ref in blocks:
            def in_block(q, m):
                c = cells.get((q, m))
                return c is not None and (not b or (b == "odometry") == bool(c.get("odo") or m in ODO_OF))
            ms = [m for m in ms if any(in_block(s["seq"], m) and cells[(s["seq"], m)]["status"] != "none" for s in gs)]
            wins = {m: 0 for m in ms}
            for s in gs:
                lo = {m: cells[(s["seq"], m)]["ate"] for m in ms if m not in IMU and in_block(s["seq"], m)
                      and cells[(s["seq"], m)]["status"] == "ok" and np.isfinite(cells[(s["seq"], m)]["ate"])}
                if lo:
                    wins[min(lo, key=lo.get)] += 1
            for m in ms:
                have = [s for s in gs if in_block(s["seq"], m) and cells[(s["seq"], m)]["status"] != "none"]
                ratio = lambda k: [cells[(s["seq"], m)][k] / cells[(s["seq"], ref)][k] for s in have
                                   if (s["seq"], ref) in cells and cells[(s["seq"], ref)]["status"] == "ok" and cells[(s["seq"], m)]["status"] == "ok"
                                   and all(np.isfinite(cells[(s["seq"], x)][k]) and cells[(s["seq"], x)][k] > 0 for x in (m, ref))]
                gm = lambda r: f"{math.exp(st.mean(math.log(x) for x in r)):.2f} ({len(r)})" if r else "n/a"
                fails = sum(cells[(s["seq"], m)]["fail"] for s in have)
                crashes = sum(cells[(s["seq"], m)]["status"] == "crash" for s in have)
                sc = [cells[(s["seq"], m)]["score"] for s in have if np.isfinite(cells[(s["seq"], m)].get("score", np.nan))]
                L.append(f"| {g}{' ' + b if b else ''} | {m} | {len(have)} | {gm(ratio('ate'))} | "
                         f"{gm(ratio('rte')) if g not in ('NTU VIRAL', 'Hilti 2021') else 'n/a'} | {wins[m] if m not in IMU else '–'} | {fails} ({crashes}) | "
                         f"{f'{st.mean(sc):.1f}' if sc else '–'} |")
    L.append("")

    # ---- RPE 1 m, hand-held
    hh = [s for s in S if s["group"] in ("NCD", "Oxford Spires") or (s["group"] == "Hilti 2022" and s["dense"])]
    ms = [m for m in LIDAR + IMU if any((s["seq"], m) in cells and np.isfinite(cells[(s["seq"], m)].get("rpe_t", np.nan)) for s in hh)]
    L += ["## RPE 1 m — hand-held only (translation cm / rotation °)", "",
          "evo RPE over 1 m, all pairs (evaluate_official.py); dense Hilti 2022 references (exp14 / 16 / 18) on the IMU-frame trajectory. Not on the cars (#243 / #245), "
          "not on NTU (prism positions only). **Bold** = best LiDAR-only.", "",
          "| sequence | " + " | ".join(ms) + " |", "|---" * (len(ms) + 1) + "|"]
    for s in hh:
        lo = [m for m in ms if m not in IMU]
        best = {k: best_of(cells, s["seq"], ms, k, lo) for k in ("rpe_t", "rpe_r")}
        cs = []
        for m in ms:
            c = cells.get((s["seq"], m))
            if c is None or c["status"] in ("none",):
                cs.append("–"); continue
            if c["status"] == "crash":
                cs.append("✗ crash"); continue
            t = []
            for k in ("rpe_t", "rpe_r"):
                x = val(c, k, sd=False)
                t.append("n/a" if x is None else f"**{x}**" if best[k] is not None and abs(c[k] - best[k]) < 1e-12 else x)
            cs.append(" / ".join(t))
        L.append(f"| {s['seq']}{' *special*' if s['special'] else ''} | " + " | ".join(cs) + " |")
    L.append("")
    for title, pool in [("all hand-held sequences above, special excluded", [s for s in hh if not s["special"]]),
                        ("NCD 10 + Oxford Spires 6 — the set of docs/results.md §1", [s for s in hh if not s["special"] and
                                                                                      (s["group"] == "NCD" or s["arms"] is ARMS42)])]:
        L += [f"Medians over {title} (per-sequence ratios to KISS-SLAM):", "",
              "| method | sequences | median RPE t cm | median RPE r ° | median t / KISS | median r / KISS |", "|---|---|---|---|---|---|"]
        for m in ms:
            have = [s for s in pool if (s["seq"], m) in cells and cells[(s["seq"], m)]["status"] == "ok"
                    and np.isfinite(cells[(s["seq"], m)]["rpe_t"]) and np.isfinite(cells[(s["seq"], "KISS-SLAM")]["rpe_t"])]
            if not have:
                continue
            g = lambda s, k: cells[(s["seq"], m)][k]
            kk = lambda s, k: cells[(s["seq"], "KISS-SLAM")][k]
            L.append(f"| {m} | {len(have)} | {st.median(g(s, 'rpe_t') for s in have):.2f} | {st.median(g(s, 'rpe_r') for s in have):.3f} | "
                     f"{st.median(g(s, 'rpe_t') / kk(s, 'rpe_t') for s in have):.2f} | {st.median(g(s, 'rpe_r') / kk(s, 'rpe_r') for s in have):.2f} |")
        L.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text("\n".join(L) + "\n")
    keys = list(rows[0].keys())
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---- coverage report
    C = ["# Coverage report (#248): runs found per (sequence, method) cell; ✗ = crashed run (no trajectory)", ""]
    warn = []
    for s in S:
        parts = []
        # a GT time span longer than the recording itself shows as the same low cover for every run of the sequence: one line
        cov = [x for (q, _), c in cells.items() if q == s["seq"] for x in c.get("covers", [])]
        alike = bool(cov) and max(cov) < 0.95 and max(cov) - min(cov) < 0.01
        if alike:
            warn.append(f"COVER<0.95 {s['group']} / {s['seq']}: every run ({len(cov)}) covers {min(cov):.3f}-{max(cov):.3f} of the GT time span "
                        "- the same for all methods: the GT spans beyond the recording / the run" + (" (3000-scan cut by design)" if "3000" in s["seq"] else ""))
        for m in dict.fromkeys(LIDAR + IMU + ODO):
            c = cells.get((s["seq"], m))
            if c is None or (not c["runs"] and not c["crash"] and not c["warn"]):
                continue
            parts.append(f"{m} {c['n']}" + (f"+{len(c['crash'])}✗" if c["crash"] else ""))
            for x in c["crash"]:
                warn.append(f"CRASH      {s['group']} / {s['seq']} / {m}: {x}")
            for x in ([] if alike else c["cover_low"]):
                warn.append(f"COVER<0.95 {s['group']} / {s['seq']} / {m}: {x}")
            for x in c["warn"] + c.get("errors", []):
                warn.append(f"NOTE       {s['group']} / {s['seq']} / {m}: {x}")
            if m in ("C (ours)", "C odo. (ours)") and c["n"] != 4 and not (s["group"] == "Boreas" and c["n"] == 1):
                warn.append(f"SEEDS      {s['group']} / {s['seq']} / {m}: {c['n']} runs ({', '.join(c['runs'])})")
        C.append(f"{s['group']:<14} {s['seq']:<38} " + ", ".join(parts))
    C += ["", f"{len(warn)} notes:"] + warn
    (OUT / f"{name}_coverage.txt").write_text("\n".join(C) + "\n")
    print("\n".join(C))
    print(f"\n→ {OUT / (name + '.md')}, {OUT / (name + '.csv')}, {OUT / (name + '_coverage.txt')}")


if __name__ == "__main__":
    main()
