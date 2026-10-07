# Main training and evaluation

Run commands from the RAVEL root so the relative dataset and weight paths in
the curated configs resolve correctly. Add code/main to PYTHONPATH when
calling modules directly.

- Direct Top-4 evaluation: ../ablation/llava_reid_top4_64.yaml
- Selector evaluation: config/llava_reid_selector_64.yaml
- Luna evaluation: config/gpt56_luna_top4.yaml
- SFT cold start: config/train_question.yaml
- 3K RL: dataset/rl-3k/state-pool-3k.jsonl
- 5K RL: dataset/rl-5k/state-pool-5k.jsonl

The trainer accepts --states and --limit-states, so the pool size is explicit
rather than hidden in the code. The curated trainer uses the
reward defined by the main experiment.
