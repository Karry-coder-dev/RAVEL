# RAVEL

This directory is the curated project snapshot for Retrieval-Aware Verbal
Evidence Learning for Interactive Person Re-Identification. The source tree is self-contained; large datasets, base models, and
retriever checkpoints are configured as external assets.

## Environment setup

RAVEL is intended to run on Linux with an NVIDIA GPU. Use Python 3.10 or
3.11 and install a PyTorch build that matches the CUDA driver on the target
machine. PyTorch and CUDA are intentionally not pinned here because the
correct wheel depends on the machine.

Create and activate an isolated environment, then install PyTorch from the
official selector for the target CUDA version:

```bash
conda create -n ravel python=3.10 -y
conda activate ravel

# Install torch and torchvision using the command selected for your CUDA version.
# Then install the RAVEL Python dependencies:
python -m pip install \
  accelerate bitsandbytes datasets deepspeed \
  transformers peft safetensors tokenizers huggingface_hub \
  numpy scipy scikit-learn pillow pyyaml tqdm einops timm \
  ftfy regex prettytable shortuuid rouge \
  openai httpx requests qwen-vl-utils wandb tyro
```

The repository vendors the project-side LLaVA and TRL code, so they do not
need to be installed as separate packages. flash-attn is optional: the
curated configs use SDPA by default. Install it separately only when
attn_implementation: flash_attention_2 is selected.

Run commands from the repository root and expose the main package directory:

```bash
cd /path/to/RAVEL
conda activate ravel
export PYTHONPATH="$PWD/code/main:$PYTHONPATH"
export RAVEL_PYTHON=python
```

Check the environment before launching a long job:

```bash
python - <<'PY'
import sys
import torch
import transformers
import accelerate
import peft

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
PY
python -m compileall -q code
```

The main RAVEL SFT and RL scripts are configured for one visible training
GPU. code/main/train-question.sh uses one torchrun process, and
code/main/run-ravel-main.sh defaults to CUDA_VISIBLE_DEVICES=0. Keep the
training GPU separate from the GPU running the answerer service when both
are active:

```bash
export CUDA_VISIBLE_DEVICES=0
```

The answerer is accessed through an OpenAI-compatible SGLang endpoint. Start
the answerer service separately on another GPU, expose port 10500, and keep
the endpoint running before SFT, RL, or evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path <answerer-model> \
  --host 0.0.0.0 \
  --port 10500
```

The YAML files use http://localhost:10500/v1 and the API key Qwen-7B by
default. Verify the endpoint before running RAVEL:

```bash
curl http://localhost:10500/v1/models
```

The usual execution order is:

1. Start the SGLang answerer service.
2. Run supervised cold start with code/main/train-question.sh if the SFT
   adapter is not already available.
3. Run the three-thousand-state RL experiment with
   code/main/run-ravel-main.sh.
4. Run evaluation with the selected YAML configuration.

If torch.compile fails during SFT, set torch_compile: false in the SFT
configuration. If a module is not found, check that the environment is
activated and that PYTHONPATH points to code/main. If CUDA memory is
insufficient, reduce the configured batch size or group size before changing
the model code.

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

`````bash
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
