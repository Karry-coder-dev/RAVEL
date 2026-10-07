#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${RAVEL_PYTHON:-python}"
EXP=ravel-main
LOGDIR=dataset/evaluation-records/rl-logs
ANALYSIS=dataset/evaluation-records/rl-analysis

mkdir -p "$LOGDIR" "$ANALYSIS" "weight/$EXP"

printf '[%s] training started\n' "$EXP" | tee "$LOGDIR/train_${EXP}.log"
"$PY" code/main/rl_train_questioner_online_grpo_validity.py \
  --config code/main/config/baseline.yaml \
  --llava-config code/main/config/train_question.yaml \
  --states dataset/rl-3k/state-pool-3k.jsonl \
  --output-dir "weight/$EXP/adapter" \
  --metrics-out "$ANALYSIS/${EXP}_metrics.jsonl" \
  --samples-out "$ANALYSIS/${EXP}_samples.jsonl" \
  --limit-states 3000 \
  --group-size 8 \
  --grad-accum-states 2 \
  --epochs 1 \
  --lr 1e-6 \
  --beta 0.03 \
  --clip-eps 0.2 \
  --repeat-coef 0.02 \
  --invalid-reward -0.2 \
  --rank1-entry-bonus 0.35 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 100 \
  --seed 42 \
  --log-steps 10 2>&1 | tee -a "$LOGDIR/train_${EXP}.log"

printf '[%s] merge started\n' "$EXP" | tee "$LOGDIR/merge_${EXP}.log"
"$PY" code/main/merge_rl_adapter_for_eval.py \
  --base "weight/reference-model" \
  --adapter "weight/$EXP/adapter" \
  --out "weight/$EXP/merged" 2>&1 | tee -a "$LOGDIR/merge_${EXP}.log"

printf '[%s] evaluation started\n' "$EXP" | tee "$LOGDIR/eval_${EXP}.log"
"$PY" code/main/main_eval.py --config_file "code/main/config/${EXP}.yaml" \
  2>&1 | tee -a "$LOGDIR/eval_${EXP}.log"
printf '[%s] all complete\n' "$EXP" | tee -a "$LOGDIR/eval_${EXP}.log"
