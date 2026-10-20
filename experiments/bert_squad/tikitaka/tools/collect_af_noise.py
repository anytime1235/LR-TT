#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Collect the asymmetry-scale (r_AF) and noise-scale (r_noise) sweep points
of run_af_noise_tuned_sweep.sh into plottable tables (Fig. 6c, Supp. Fig. S8).

One row per (curve, r_AF, r_noise). Each curve is the directory of one anchor,
i.e. one set of frozen hyperparameters (the best trial of a TPE study):

  AF0nz0  tiki_af_noise_tuned_20260809/af0nz0
          ttv1_auxbit10b_anchor87 (r_AF 0, r_noise 0)
  AF1nz0  tiki_af_noise_tuned_20260809/af1nz0
          ttv1_6t1c_af1_qkvo (r_AF 1, r_noise 0)
  AF1nz1  tiki_af_noise_tuned_20260809/af1nz1
          ttv1_6t1c_full_qkvo (r_AF 1, r_noise 1)

Writes SWEEP.csv (one row per sweep point with F1, EM, the hyperparameters and
the I/O settings of its config.json), epoch_metrics.csv (one row per evaluated
epoch), PLOT1_af_axis.csv (best_f1 over r_AF at r_noise 0, one column per
curve) and PLOT2_noise_axis.csv (best_f1 over r_noise at r_AF 1), and copies
the PROVENANCE.txt of each anchor directory.

Every point is a single 5-epoch run with --patience 0. best_f1 is the highest
per-epoch F1 (used by the pivots), final_f1 the F1 after the last epoch;
complete and epochs_evaluated show whether a run finished all epochs. Each
call rewrites the tables.

usage: collect_af_noise.py [studies_root] [out_dir]

studies_root defaults to ../results (the RESULTS_ROOT default of the
launchers); out_dir defaults to <studies_root>/exports/ttv1_af_noise_sweeps.
"""
import csv, glob, json, os, re, shutil, sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    print(__doc__)
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(PAPER, "exports", "ttv1_af_noise_sweeps")

TUNED = f"{PAPER}/tiki_af_noise_tuned_20260809"
SOURCES = [
    ("AF0nz0", f"{TUNED}/af0nz0"),
    ("AF1nz0", f"{TUNED}/af1nz0"),
    ("AF1nz1", f"{TUNED}/af1nz1"),
]
# curves of the PLOT*.csv pivots
PLOT_CURVES = ("AF0nz0", "AF1nz0", "AF1nz1")

HP = ("fast_lr", "transfer_lr", "transfer_every", "w_max_fast",
      "classifier_lr", "ln_lr", "analog_lr", "dw_min", "n_bits_slow",
      "device_type", "device_type_slow", "target_layers", "patience", "epochs")

TAG = re.compile(r"^af([\dp]+)_nz([\dp]+)$")


def num(s):
    return float(s.replace("p", "."))


rows, epochs = [], []
for curve, root in SOURCES:
    for d in sorted(glob.glob(os.path.join(root, "af*_nz*"))):
        m = TAG.match(os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        cfg_p = os.path.join(d, "config.json")
        if not os.path.exists(cfg_p):
            continue
        cfg = json.load(open(cfg_p))
        eps = []
        log_p = os.path.join(d, "training_log.csv")
        if os.path.exists(log_p):
            eps = [r for r in csv.DictReader(open(log_p)) if r.get("f1")]
        done = os.path.exists(os.path.join(d, "summary.json"))
        last = eps[-1] if eps else {}
        best = max(eps, key=lambda r: float(r["f1"])) if eps else {}
        row = {
            "curve": curve, "tag": os.path.basename(d),
            "af": num(m.group(1)), "noise": num(m.group(2)),
            "complete": done, "epochs_evaluated": len(eps),
            "final_f1": float(last["f1"]) if last else None,
            "final_em": float(last["em"]) if last else None,
            "best_f1": float(best["f1"]) if best else None,
            "best_em": float(best["em"]) if best else None,
            "best_f1_epoch": int(best["epoch"]) if best else None,
        }
        # the device axes must agree with the directory name; report if not
        row["af_cfg"] = cfg.get("ls_gamma_up_ratio")
        row["noise_cfg"] = cfg.get("ls_noise_ratio")
        row["axis_mismatch"] = (row["af_cfg"] != row["af"]
                                or row["noise_cfg"] != row["noise"])
        for k in HP:
            row[k] = cfg.get(k)
        io = cfg.get("io_resolved", {}).get("forward", {})
        row["io_out_noise"] = io.get("out_noise")
        row["io_nm"] = io.get("noise_management")
        row["io_bm"] = io.get("bound_management")
        row["run_dir"] = d
        rows.append(row)
        for r in eps:
            epochs.append({"curve": curve, "tag": os.path.basename(d),
                           "af": row["af"], "noise": row["noise"],
                           "epoch": r["epoch"], "step": r["step"],
                           "f1": r["f1"], "em": r["em"]})

if not rows:
    sys.exit(f"no af*_nz* run directories found under {PAPER}")
rows.sort(key=lambda r: (r["curve"], r["noise"], r["af"]))
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "SWEEP.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
if epochs:
    with open(os.path.join(OUT, "epoch_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(epochs[0].keys()))
        w.writeheader(); w.writerows(epochs)

for curve, root in SOURCES:
    p = os.path.join(root, "PROVENANCE.txt")
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, f"PROVENANCE_{curve.replace('@','_')}.txt"))

# --- plot-ready views: one file per plot, curves side by side ---------------
def pivot(fixed_key, fixed_val, axis, fname):
    curves = [c for c in PLOT_CURVES
              if any(r["curve"] == c and r[fixed_key] == fixed_val
                     and r["best_f1"] is not None for r in rows)]
    xs = sorted({r[axis] for r in rows
                 if r["curve"] in curves and r[fixed_key] == fixed_val
                 and r["best_f1"] is not None})
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([axis] + curves)
        for x in xs:
            line = [x]
            for cv in curves:
                hit = [r for r in rows if r["curve"] == cv
                       and r[fixed_key] == fixed_val and r[axis] == x
                       and r["best_f1"] is not None]
                line.append(round(hit[0]["best_f1"], 4) if hit else "")
            w.writerow(line)
    return curves, xs

c1, x1 = pivot("noise", 0.0, "af", "PLOT1_af_axis.csv")
c2, x2 = pivot("af", 1.0, "noise", "PLOT2_noise_axis.csv")

print(f"{'curve':<12}{'AF':>6}{'noise':>7}{'F1':>8}{'EM':>8}{'ep':>4}  ok")
for r in rows:
    print(f"{r['curve']:<12}{r['af']:>6}{r['noise']:>7}"
          f"{(r['best_f1'] or 0):>8.2f}{(r['best_em'] or 0):>8.2f}"
          f"{r['epochs_evaluated']:>4}  {'Y' if r['complete'] else 'n'}"
          + ("   <<< AXIS MISMATCH" if r["axis_mismatch"] else ""))
print(f"\nplot 1 (x=AF, noise=0)    curves={c1}  x={x1}")
print(f"plot 2 (x=noise, AF=1)    curves={c2}  x={x2}")
print(f"\n-> {OUT}")
