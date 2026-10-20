#!/usr/bin/env bash
# (C) Copyright 2025, 2026 Young Woon Cho, Jaeweon Park, Jongun Won and Sangbum Kim,
# Seoul National University. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Run fine_bert_squad_lrtt.py for a set of configs, one after the other, in the foreground
# on one GPU.
#
#   GPU=0 ./run_fine_configs.sh                          # every config under configs/
#   GPU=1 ./run_fine_configs.sh rank_sweep_onehot t86    # one or more config groups
#   ./run_fine_configs.sh figS9/fullC_r32.json           # single configs (relative to configs/)
#
# Config groups and paper panels (README_fine.md gives them per file pattern):
#   rank_sweep_onehot  Fig. 5i, Supp. Fig. S10
#   rank_sweep_set     Fig. 6b (ideal transfer)
#   figS9              Figs. 5f, 6c, 6d; Supp. Figs. S7, S8a, S9a, S11; Supp. Section S8
#   t86                Fig. 6c; Supp. Figs. S8a, S8b, S9b
#   fig6c              Supp. Fig. S8a
#   fig6c2             Supp. Fig. S8b
#   s9ext              Supp. Figs. S9a, S9b
#
# The outputs of a run go to $OUT_DIR/<group>/ (default: results/BERT_SQUAD_LRTT_FINE next
# to this script), its console output to $OUT_DIR/<group>/runlog_<name>.txt. PYTHON selects
# the interpreter (default: python). The script stops at the first run that exits with an
# error.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-0}
OUT_DIR=${OUT_DIR:-$HERE/results/BERT_SQUAD_LRTT_FINE}
PYTHON=${PYTHON:-python}

configs=()
if [ $# -eq 0 ]; then
    mapfile -t configs < <(find "$HERE/configs" -name '*.json' | sort)
else
    for arg in "$@"; do
        if [ -d "$HERE/configs/$arg" ]; then
            mapfile -t -O "${#configs[@]}" configs < <(find "$HERE/configs/$arg" -name '*.json' | sort)
        elif [ -f "$HERE/configs/$arg" ]; then
            configs+=("$HERE/configs/$arg")
        elif [ -f "$arg" ]; then
            configs+=("$arg")
        else
            echo "unknown config or group: $arg" >&2
            exit 1
        fi
    done
fi

echo "${#configs[@]} config(s) on GPU $GPU"
for cfg in "${configs[@]}"; do
    group=$(basename "$(dirname "$cfg")")
    name=$(basename "$cfg" .json)
    mkdir -p "$OUT_DIR/$group"
    log="$OUT_DIR/$group/runlog_$name.txt"
    echo "=== $cfg -> $log"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$HERE/fine_bert_squad_lrtt.py" --config "$cfg" \
        --output-dir "$OUT_DIR/$group" 2>&1 | tee "$log"
done
