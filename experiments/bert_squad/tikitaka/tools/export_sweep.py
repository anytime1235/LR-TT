#!/usr/bin/env python3
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Export a paper_experiment.py sweep (fixed-mode runs and/or Optuna TPE studies).

Handles both layouts by walking for config.json:
  fixed mode : <root>/<run>/config.json + training_log.csv [+ summary.json]
  tpe mode   : <root>/<study>/trial_XXXX/config.json  (+ <root>/db/<study>.db)

Writes runs.csv (one row per run directory with hyperparameters, F1 and EM),
epoch_metrics.csv (one row per evaluated epoch) and runs.json (the rows plus
config.json, summary.json and the per-epoch metrics), and copies the study
databases and the top-level *.log and pids.txt files of <sweep_root> into
<out_dir>.

best_em is the EM of the best-F1 epoch in training_log.csv. transfer_every,
which Optuna stores as an index into its choices, is decoded and checked
against config.json.

usage: export_sweep.py <sweep_root> <out_dir> [label]
"""
import csv, json, os, sqlite3, sys, datetime, shutil, glob


def read_epochs(run_dir):
    """Per-epoch eval rows (rows where f1 is populated) from training_log.csv."""
    p = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return [{k: (v if v != "" else None) for k, v in r.items()}
                for r in csv.DictReader(f) if r.get("f1")]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def db_params(root):
    """{(study, trial_number): {param: decoded_value}} plus trial states/values."""
    params, meta = {}, {}
    for db in sorted(glob.glob(os.path.join(root, "db", "*.db"))):
        study = os.path.splitext(os.path.basename(db))[0]
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        num = {r["trial_id"]: r["number"] for r in cur.execute("select trial_id, number from trials")}
        for r in cur.execute("select trial_id, number, state, datetime_start, datetime_complete from trials"):
            meta[(study, r["number"])] = (r["state"], r["datetime_start"], r["datetime_complete"])
        for r in cur.execute("select trial_id, value from trial_values"):
            k = (study, num[r["trial_id"]])
            params.setdefault(k, {})["_objective"] = r["value"]
        for r in cur.execute("select trial_id, param_name, param_value, distribution_json from trial_params"):
            v = r["param_value"]
            d = read_json_str(r["distribution_json"])
            if d.get("name") == "CategoricalDistribution":
                v = d["attributes"]["choices"][int(round(v))]   # index -> value
            params.setdefault((study, num[r["trial_id"]]), {})[r["param_name"]] = v
        con.close()
    return params, meta


def resolve_studies(root, cfg_paths, dbp):
    """{trial_dir_name: db_study_name} decided by hyperparameter agreement.

    The output directory of a study and its Optuna study name can differ, and
    one db/ directory can hold several studies. A directory name that is part
    of exactly one study name settles the pairing. Otherwise every candidate
    study is scored by how often its recorded parameters equal the values in
    the trials' own config.json, and the best-scoring study is taken. Ties, no
    overlap, or agreement below 90% leave the directory unmapped, and its
    database columns stay blank.
    """
    groups = {}
    for p in cfg_paths:
        run_dir = os.path.dirname(p)
        if not os.path.basename(run_dir).startswith("trial_"):
            continue
        groups.setdefault(os.path.basename(os.path.dirname(run_dir)), []).append(
            (int(os.path.basename(run_dir).split("_")[1]), p))

    studies = sorted({k[0] for k in dbp})
    out, ambiguous = {}, []
    for gdir, trials in sorted(groups.items()):
        # A dir name that names exactly one study settles it: sibling studies
        # such as qkvo/ffn share a search space, so their params can agree
        # equally well and scoring alone would not separate them.
        named = [st for st in studies if gdir in st]
        if len(named) == 1:
            out[gdir] = named[0]
            continue
        scores = {}
        for st in studies:
            hit = tot = 0
            for trial, cfg_path in trials:
                p = dbp.get((st, trial))
                if not p:
                    continue
                cfg = read_json(cfg_path)
                for name, dbv in p.items():
                    if name.startswith("_") or name not in cfg:
                        continue
                    tot += 1
                    try:
                        hit += abs(float(dbv) - float(cfg[name])) <= 1e-6 * max(
                            1.0, abs(float(dbv)))
                    except (TypeError, ValueError):
                        hit += str(dbv) == str(cfg[name])
            scores[st] = (hit / tot, tot) if tot else (0.0, 0)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0]))
        if not ranked or ranked[0][1][1] == 0:
            ambiguous.append((gdir, "no db overlap", scores))
            continue
        best, rest = ranked[0], ranked[1:]
        if rest and rest[0][1][0] >= best[1][0]:
            ambiguous.append((gdir, "tie", scores))
            continue
        if best[1][0] < 0.9:
            # Below 90% agreement the study of this dir is taken to be absent
            # from db/; the dir stays unmapped and its db columns stay blank.
            ambiguous.append((gdir, f"no study agrees (best {best[0]} at "
                                    f"{best[1][0]:.0%}) - db columns left blank",
                              scores))
            continue
        out[gdir] = best[0]
        if best[1][0] < 1.0:
            ambiguous.append((gdir, f"best study agrees only "
                                    f"{best[1][0]:.0%} ({best[0]})", scores))
    return out, ambiguous


def read_json_str(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


def main():
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(root)
    os.makedirs(out, exist_ok=True)

    dbp, dbmeta = db_params(root)

    cfg_paths = sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True))
    dir2study, ambiguous = resolve_studies(root, cfg_paths, dbp)

    rows, epoch_rows, full = [], [], []
    mismatch = []
    for cfg_path in cfg_paths:
        run_dir = os.path.dirname(cfg_path)
        rel = os.path.relpath(run_dir, root)
        cfg = read_json(cfg_path)
        summ = read_json(os.path.join(run_dir, "summary.json")).get("results", {})
        eps = read_epochs(run_dir)

        base = os.path.basename(run_dir)
        if base.startswith("trial_"):
            study, trial = os.path.basename(os.path.dirname(run_dir)), int(base.split("_")[1])
            # The output dir name and the Optuna study name can differ, and
            # trial numbers repeat across studies sharing one db dir;
            # resolve_studies() pairs each trial dir with its study.
            db_study = dir2study.get(study)
            key = (db_study, trial) if db_study else None
            p = dbp.get(key, {}) if key else {}
            state, dt0, dt1 = dbmeta.get(key, ("", None, None)) if key else ("", None, None)
        else:
            study, trial, p, state = rel, None, {}, "FIXED"
            dt0 = dt1 = None

        f1s = [(int(float(e["epoch"])), float(e["f1"]), float(e["em"])) for e in eps]
        best = max(f1s, key=lambda x: x[1]) if f1s else None
        fin = f1s[-1] if f1s else None

        # config.json is authoritative for the actually-applied values
        te_cfg = cfg.get("transfer_every")
        if te_cfg is not None and p.get("transfer_every") is not None:
            if int(round(float(te_cfg))) != int(round(float(p["transfer_every"]))):
                mismatch.append((rel, p["transfer_every"], te_cfg))

        dur = None
        if dt0 and dt1:
            dur = round((datetime.datetime.fromisoformat(dt1)
                         - datetime.datetime.fromisoformat(dt0)).total_seconds() / 60, 1)
        elif summ.get("wall_time_s"):
            dur = round(summ["wall_time_s"] / 60, 1)

        row = {
            "label": label, "run": rel, "group": study, "trial": trial, "state": state,
            "method": cfg.get("method"), "target_layers": cfg.get("target_layers"),
            "seed": cfg.get("seed"),
            "n_bits": cfg.get("n_bits"), "n_bits_slow": cfg.get("n_bits_slow"),
            "dw_min": cfg.get("dw_min"), "w_max_fast": cfg.get("w_max_fast"),
            "fast_lr": cfg.get("fast_lr"),
            "analog_lr": cfg.get("analog_lr"), "classifier_lr": cfg.get("classifier_lr"),
            "ln_lr": cfg.get("ln_lr"), "transfer_lr": cfg.get("transfer_lr"),
            "transfer_every": te_cfg,
            "best_f1": best[1] if best else None, "best_em": best[2] if best else None,
            "best_f1_epoch": best[0] if best else None,
            "final_f1": fin[1] if fin else None, "final_em": fin[2] if fin else None,
            "summary_best_f1": summ.get("best_f1"),
            "objective_f1": p.get("_objective"),
            "epochs_evaluated": len(f1s), "duration_min": dur,
            "io_bits": cfg.get("io_bits"),
            "io_out_noise": (cfg.get("io_resolved") or {}).get("forward", {}).get("out_noise"),
            "io_nm": (cfg.get("io_resolved") or {}).get("forward", {}).get("noise_management"),
            "io_bm": (cfg.get("io_resolved") or {}).get("forward", {}).get("bound_management"),
            "run_dir": run_dir,
        }
        rows.append(row)
        full.append({**row, "config": cfg, "summary": summ,
                     "epoch_metrics": [{"epoch": e[0], "f1": e[1], "em": e[2]} for e in f1s]})
        for e in eps:
            epoch_rows.append({"label": label, "run": rel, "trial": trial,
                               "n_bits": cfg.get("n_bits"), "n_bits_slow": cfg.get("n_bits_slow"),
                               "epoch": int(float(e["epoch"])), "step": e["step"], "loss": e["loss"],
                               "f1": e["f1"], "em": e["em"], "wall_time_s": e.get("wall_time_s")})

    if not rows:
        print(f"[export] no config.json under {root} - nothing to write")
        return

    rows.sort(key=lambda r: (str(r["group"]), r["trial"] if r["trial"] is not None else -1))
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # a sweep where every run died before its first eval has no epoch rows; still
    # write the header so the file exists and the empty run set is visible
    EPOCH_COLS = ["label", "run", "trial", "n_bits", "n_bits_slow", "epoch",
                  "step", "loss", "f1", "em", "wall_time_s"]
    with open(os.path.join(out, "epoch_metrics.csv"), "w", newline="") as f:
        cols = list(epoch_rows[0].keys()) if epoch_rows else EPOCH_COLS
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(epoch_rows)
    with open(os.path.join(out, "runs.json"), "w") as f:
        json.dump(full, f, indent=2)
    for db in glob.glob(os.path.join(root, "db", "*.db")):
        shutil.copy2(db, os.path.join(out, os.path.basename(db)))
    for log in glob.glob(os.path.join(root, "*.log")) + glob.glob(os.path.join(root, "pids.txt")):
        shutil.copy2(log, os.path.join(out, os.path.basename(log)))

    print(f"[export] {label}: {len(rows)} runs, {len(epoch_rows)} epoch rows -> {out}")
    for gdir, study in sorted(dir2study.items()):
        print(f"    dir {gdir:<24} -> study {study}")
    for gdir, why, scores in ambiguous:
        print(f"[export] WARNING study mapping for dir {gdir}: {why} - {scores}")
    if mismatch:
        print(f"[export] WARNING transfer_every mismatch (db vs config.json): {mismatch}")
    done = [r for r in rows if r["best_f1"] is not None]
    for r in sorted(done, key=lambda r: -r["best_f1"])[:8]:
        print(f"    {r['run']:<28} n_bits={r['n_bits']} slow={r['n_bits_slow']} "
              f"F1={r['best_f1']:.2f} EM={r['best_em']:.2f} alr={r['analog_lr']:.6g} clr={r['classifier_lr']:.6g}")


if __name__ == "__main__":
    main()
