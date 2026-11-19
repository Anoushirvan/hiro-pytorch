#!/usr/bin/env bash
# Step 7: build every figure/GIF for the report from the Step 6 results (run on the Mac after copying them back).
#
#   1. copy results from the PC (copies files only):
#        rsync -az <user>@<pc-host>:<path>/hiro-pytorch/results/runs/ results/runs/
#        rsync -az <user>@<pc-host>:<path>/hiro-pytorch/results/checkpoints/ results/checkpoints/
#   2. bash scripts/make_step7_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/step7
mkdir -p "$OUT"

# learning curves + summary table
python scripts/plot_step6.py --root results/runs --out results/step6

# best seed per task (highest mean success over the last 3 evaluations)
best_seed() {   # $1 = env, $2 = variant, $3 = success column
  python - "$1" "$2" "$3" <<'EOF'
import csv, glob, sys, os
env, var, col = sys.argv[1:]
best, arg = -1, None
for f in sorted(glob.glob(f"results/runs/{env}/{var}/seed*.csv")):
    rows = list(csv.DictReader(open(f)))
    if rows and col in rows[0]:
        v = sum(float(r[col]) for r in rows[-3:]) / len(rows[-3:])
        if v > best:
            best, arg = v, os.path.basename(f)[4:-4]
print(arg if arg is not None else 0)
EOF
}

for spec in "AntMaze:0_16" "AntPush:0_19" "AntFall:0_27"; do
  env=${spec%%:*}; tgt=${spec#*:}
  [[ -d results/checkpoints/$env/hiro ]] || { echo "skip $env (no checkpoints)"; continue; }
  # subgoal plots: every HIRO seed side by side, plus the no-correction ablation for Ant Maze
  ckpts=(results/checkpoints/$env/hiro/seed*.pt)
  labels=(); for c in "${ckpts[@]}"; do labels+=("HIRO $(basename "$c" .pt)"); done
  python scripts/viz_goals.py --env "$env" --ckpt "${ckpts[@]}" --labels "${labels[@]}" --out "$OUT/${env}_goals_hiro.png"
  if [[ -d results/checkpoints/$env/hiro_no_correction ]]; then
    ckpts=(results/checkpoints/$env/hiro_no_correction/seed*.pt)
    labels=(); for c in "${ckpts[@]}"; do labels+=("no correction $(basename "$c" .pt)"); done
    python scripts/viz_goals.py --env "$env" --ckpt "${ckpts[@]}" --labels "${labels[@]}" \
      --out "$OUT/${env}_goals_no_correction.png"
  fi
  s=$(best_seed "$env" hiro "success_$tgt")
  python scripts/make_gif.py --env "$env" --ckpt "results/checkpoints/$env/hiro/seed$s.pt" --out "$OUT/${env}_hiro_seed$s.gif"
done
echo "figures in results/step6 and $OUT"
