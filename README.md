# RAVEL

This directory is the curated project snapshot for Retrieval-Aware Verbal
Evidence Learning for Interactive Person Re-Identification. The source tree is self-contained; large datasets, base models, and
retriever checkpoints are configured as external assets.

## Layout

- code/: training, evaluation, motivation, behavior-analysis, transfer, and
  ablation code.
- dataset/: documentation and mount points for external data assets. The
  actual Interactive-PEDES files, RL state pools, Qwen3-VL audit records, and
  human-review records are not tracked in this repository.
- weight/: external model and retriever checkpoint mount points. Weight files
  are intentionally not tracked.

The public source tree contains code and lightweight documentation only. Raw
images, benchmark annotations, RL pools, audit records, human annotations,
base LLMs, and model checkpoints must be downloaded separately and placed at
the paths described in dataset/README.md and the configuration files.

## Reproduction switches

code/ablation/llava_reid_top4_64.yaml uses direct Top-4 candidates.
code/main/config/llava_reid_selector_64.yaml uses the selector protocol.
The main evaluator selects the retriever from YAML; code/ablation/ravel-rde.yaml
selects the RDE replacement with retriever_type: rde.

The RL trainer switches between the state pools through --states and
--limit-states:

- 3K: dataset/rl-3k/state-pool-3k.jsonl with --limit-states 3000
- 5K: dataset/rl-5k/state-pool-5k.jsonl with --limit-states 5000

The reward combines reciprocal-rank change,
the Rank-1 transition term, the invalid-question penalty, and the repeat penalty.
Rank-5/Rank-10 terms and regression penalties are excluded from the
curated trainer.


## External assets

The main code does not depend on sibling repositories. Large assets are supplied
separately:

Set the asset locations before training or transfer evaluation, for example:

```bash
export OPEN_SOURCE_LLM_ROOT=/path/to/open_source_llm
export RAVEL_TRANSFER_DATA_ROOT=/path/to/transfer-assets
export RAVEL_IRRA_ROOT=/path/to/IRRA-official-repro
export RAVEL_RDE_PROJECT_ROOT=/path/to/RDE-project
export RAVEL_RDE_CHECKPOINT_DIR=/path/to/RDE-checkpoint
```

- `OPEN_SOURCE_LLM_ROOT`: directory containing the LLaVA-OneVision base model
  and SigLIP vision tower referenced by `code/main/config/train_question.yaml`.
- `RAVEL_TRANSFER_DATA_ROOT`: transfer benchmark root containing `data/` and
  `RDA_data/` in the layout used by the transfer script.
- `RAVEL_IRRA_ROOT`: external IRRA repository containing the dataset-specific
  transfer checkpoints.
- `RAVEL_RDE_PROJECT_ROOT`: RDE project root containing its `model/` package.
- `RAVEL_RDE_CHECKPOINT_DIR`: directory containing the Interactive-PEDES RDE
  `configs.yaml` and `best.pth`.

The optional PlugIR baselines use `external/PlugIR/` when their configs are
selected. The main RAVEL training and evaluation path only requires the
Interactive-PEDES assets, the reference/RAVEL checkpoints, and the external
open-source base model.
