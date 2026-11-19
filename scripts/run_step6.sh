#!/usr/bin/env bash
# Step 6 launcher: all Ant experiments in parallel, detached (survive ssh logout).
#
#   bash scripts/run_step6.sh bench              # throughput test: short runs on each device, prints steps/s
#   bash scripts/run_step6.sh run                # launch all runs
#   bash scripts/run_step6.sh dry                # print the commands only
#   bash scripts/run_step6.sh status             # last log line of every run
#   bash scripts/run_step6.sh stop               # kill all launched runs
#
# Configure with environment variables, e.g.
#   STEPS=2000000 DEVICES="cpu" bash scripts/run_step6.sh run
#   STEPS=2000000 DEVICES="cuda:0 cuda:1" bash scripts/run_step6.sh run   # round-robin over GPUs
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-dry}
PY=${PY:-python}
STEPS=${STEPS:-2000000}
DEVICES=${DEVICES:-cpu}
EVAL_EVERY=${EVAL_EVERY:-50000}
EVAL_EPISODES=${EVAL_EPISODES:-5}
MAZE_SEEDS=${MAZE_SEEDS:-"0 1 2"}
PUSHFALL_SEEDS=${PUSHFALL_SEEDS:-"0 1"}
BENCH_PAR=${BENCH_PAR:-"1 6 12"}       # numbers of parallel runs to try in bench mode
LOGDIR=results/logs
PIDFILE=$LOGDIR/step6_pids.txt
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- run list: "name|command args"
jobs=()
for s in $MAZE_SEEDS; do
  jobs+=("AntMaze_hiro_s$s|scripts/train_hiro.py --env AntMaze --correction --seed $s")
  jobs+=("AntMaze_hiro_no_correction_s$s|scripts/train_hiro.py --env AntMaze --seed $s")
  # flat TD3 baseline ("No HRL"): same exploration noise as the HIRO low level (sigma 1 on +-30)
  jobs+=("AntMaze_td3_s$s|scripts/train_flat.py --env AntMaze --variant td3 --expl-noise 0.0333 --start-steps 2500 --seed $s")
done
for s in $PUSHFALL_SEEDS; do
  jobs+=("AntPush_hiro_s$s|scripts/train_hiro.py --env AntPush --correction --seed $s")
  jobs+=("AntFall_hiro_s$s|scripts/train_hiro.py --env AntFall --correction --seed $s")
done

read -r -a devs <<< "$DEVICES"

case "$MODE" in
  dry|run)
    [[ "$MODE" == run ]] && : > "$PIDFILE"
    i=0
    for j in "${jobs[@]}"; do
      name=${j%%|*}; cmd=${j#*|}
      dev=${devs[$((i % ${#devs[@]}))]}
      full="$PY $cmd --steps $STEPS --eval-every $EVAL_EVERY --eval-episodes $EVAL_EPISODES --device $dev"
      if [[ "$MODE" == dry ]]; then
        echo "$full  > $LOGDIR/step6_$name.log"
      else
        nohup $full > "$LOGDIR/step6_$name.log" 2>&1 &
        echo "$! $name" >> "$PIDFILE"
        echo "started $name (pid $!, $dev)"
      fi
      i=$((i + 1))
    done
    [[ "$MODE" == run ]] && echo "${#jobs[@]} runs started; PIDs in $PIDFILE. Check with: bash scripts/run_step6.sh status"
    ;;
  status)
    for f in "$LOGDIR"/step6_*.log; do
      printf "%-40s %s\n" "$(basename "$f" .log)" "$(grep -E '^step' "$f" | tail -1 | cut -c1-150)"
    done
    [[ -f "$PIDFILE" ]] && echo "alive: $(while read -r pid _; do kill -0 "$pid" 2>/dev/null && echo x; done < "$PIDFILE" | wc -l) / $(wc -l < "$PIDFILE")"
    ;;
  stop)
    while read -r pid name; do kill "$pid" 2>/dev/null && echo "stopped $name"; done < "$PIDFILE"
    ;;
  bench)
    # HIRO+correction is the slowest variant; measure steady-state steps/s per run and in total.
    tmp=$(mktemp -d)
    for dev in "${devs[@]}"; do
      for n in $BENCH_PAR; do
        for k in $(seq 1 "$n"); do
          $PY scripts/train_hiro.py --env AntMaze --correction --steps 8000 --start-steps 1000 \
            --eval-every 8000 --eval-episodes 1 --device "$dev" --seed "$k" \
            --out "$tmp/runs" --tag "bench_${dev}_$n" > "$tmp/${dev}_${n}_$k.log" 2>&1 &
        done
        wait
        per=$(grep -h '^step' "$tmp/${dev}_${n}"_*.log | awk '{s+=$(NF-1)} END {printf "%.0f", s/NR}')
        echo "device=$dev parallel=$n : ${per} steps/s per run, $((per * n)) steps/s total"
      done
    done
    rm -rf "$tmp" results/checkpoints/AntMaze/bench_*
    ;;
  *) echo "usage: $0 {bench|dry|run|status|stop}"; exit 1 ;;
esac
