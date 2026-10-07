# Ablations

This directory contains the candidate-input, retriever, and
training-procedure ablations.

- llava_reid_top4_64.yaml: LLaVA-ReID with direct Top-4 candidate input,
  without a selector.
- ravel-rde.yaml: RAVEL with the RDE-192 retriever. It changes
  retriever_type from irra to rde and uses the visual answerer.
- rl3k_native.yaml: 3K-state RL from the native questioner.

Run the RDE ablation through the shared evaluator:

~~~bash
python code/main/main_eval.py --config_file code/ablation/ravel-rde.yaml
~~~

Set RAVEL_RDE_PROJECT_ROOT and RAVEL_RDE_CHECKPOINT_DIR before running.
