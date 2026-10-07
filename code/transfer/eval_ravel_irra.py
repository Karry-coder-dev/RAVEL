"""Run the corrected cross-dataset transfer protocol.

The interaction uses the frozen RAVEL questioner and the main Interactive-PEDES
IRRA retriever.  Each query starts from the benchmark's original caption.  The
multimodal answerer sees the ground-truth image directly.  After five rounds,
the final interactive features are averaged with features from the frozen,
dataset-specific standard IRRA model trained on the original captions.
"""

import argparse
import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
import transformers
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


LLAVA_ROOT = Path(os.environ.get("RAVEL_ROOT", Path(__file__).resolve().parents[2]))
MAIN_ROOT = LLAVA_ROOT / "code/main"
sys.path.insert(0, str(MAIN_ROOT))
RDE_DATA_ROOT = Path(os.environ.get("RAVEL_TRANSFER_DATA_ROOT", LLAVA_ROOT / "external/transfer-data"))
IRRA_ROOT = Path(os.environ.get("RAVEL_IRRA_ROOT", LLAVA_ROOT / "external/IRRA-official-repro"))

DATASETS = {
    "CUHK-PEDES": {
        "annotations": RDE_DATA_ROOT / "RDA_data/r_cuhk_1110_dec_1110.json",
        "image_root": RDE_DATA_ROOT / "data/CUHK-PEDES/imgs",
        "irra_config": IRRA_ROOT / "run_logs_standard/CUHK-PEDES/20260910_123738_IRRA-standard/configs.yaml",
        "irra_checkpoint": IRRA_ROOT / "run_logs_standard/CUHK-PEDES/20260910_123738_IRRA-standard/best.pth",
    },
    "ICFG-PEDES": {
        "annotations": RDE_DATA_ROOT / "data/ICFG-PEDES/ICFG-PEDES.json",
        "image_root": RDE_DATA_ROOT / "data/ICFG-PEDES/imgs",
        "irra_config": IRRA_ROOT / "run_logs_standard/ICFG-PEDES/20260910_123738_IRRA-standard/configs.yaml",
        "irra_checkpoint": IRRA_ROOT / "run_logs_standard/ICFG-PEDES/20260910_123738_IRRA-standard/best.pth",
    },
}


class GalleryDataset(Dataset):
    def __init__(self, rows, image_root, transform):
        self.image_pids = [int(row["id"]) for row in rows]
        self.img_paths = [str(image_root / row["file_path"]) for row in rows]
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        with Image.open(self.img_paths[index]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return index, self.image_pids[index], tensor


class QueryDataset(Dataset):
    """Return GT image path as the answerer context and raw caption as query."""

    def __init__(self, rows, image_root):
        self.caption_pids = []
        self.gt_paths = []
        self.initial_query = []
        for row in rows:
            gt_path = str(image_root / row["file_path"])
            for caption in row["captions"]:
                self.caption_pids.append(int(row["id"]))
                self.gt_paths.append(gt_path)
                self.initial_query.append(caption)

    @property
    def fine_grained_caption(self):
        # engine_eval uses this field as the answerer context.  In this protocol
        # the context is the GT image path, not a generated memory description.
        return self.gt_paths

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        return index, self.caption_pids[index], self.gt_paths[index], self.initial_query[index]


class LimitedQueryDataset(Dataset):
    """Preserve the query metadata expected by engine_eval for smoke runs."""

    def __init__(self, base, length):
        self.base = base
        self.length = min(len(base), length)

    @property
    def fine_grained_caption(self):
        return self.base.fine_grained_caption[:self.length]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.base[index]


class StandardIRRAAdapter:
    def __init__(self, checkpoint, config_path, device):
        from model.IRRA import IRRA
        from model.clip_model import convert_weights

        with config_path.open() as handle:
            cfg = yaml.load(handle, Loader=yaml.FullLoader)
        cfg["training"] = False
        cfg["num_classes"] = 1
        args = Namespace(**cfg)
        self.text_length = int(args.text_length)
        self.device = device
        self.tokenizer = __import__("utils.simple_tokenizer", fromlist=["SimpleTokenizer"]).SimpleTokenizer()
        self.model = IRRA(args, device).to(device).eval()
        state = torch.load(checkpoint, map_location="cpu")
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(state, strict=False)
        convert_weights(self.model)
        self.model.eval()

    @torch.inference_mode()
    def encode_image(self, images):
        output = self.model.encode_image(images.to(self.device))
        if output.ndim == 3:
            output = output[:, 0, :]
        return F.normalize(output.float(), dim=-1)

    @torch.inference_mode()
    def encode_text(self, texts):
        from reid_datasets.bases import tokenize_simple

        tokens = torch.stack([
            tokenize_simple(text, self.tokenizer, self.text_length)
            for text in texts
        ]).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)


def load_benchmark(dataset_name):
    specification = DATASETS[dataset_name]
    with specification["annotations"].open() as handle:
        annotations = json.load(handle)
    rows = [row for row in annotations if row.get("split") == "test"]
    if not rows:
        raise RuntimeError(f"No test split found in {specification['annotations']}")
    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    return specification, GalleryDataset(rows, specification["image_root"], transform), QueryDataset(rows, specification["image_root"])


def load_ravel(config_path, dataset_name, interact_round):
    from model.llava_reid import LlavaForPersonReID
    from model.llava_reid_utils import AnswerVisualGeneratorSGLang
    from train_llava_reid import ModelConfig

    with config_path.open() as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config.update(config["stage_config"]["eval"])
    del config["stage_config"]
    config["stage"] = "eval"
    config["dataset_name"] = dataset_name
    config["output_dir"] = str(LLAVA_ROOT / "dataset/evaluation-records/interactive-pedes-eval-runs")
    config["interact_round"] = int(interact_round)
    parser = transformers.HfArgumentParser(ModelConfig)
    model_config = parser.parse_dict(config, allow_extra_keys=True)[0]
    logger = logging.getLogger("ravel_transfer_protocol")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    model = LlavaForPersonReID(model_config, {"model_path": model_config.question_model_path}, logger)

    # The main experiment's answerer is text-only; transfer uses the same
    # endpoint but sends the ground-truth image with each question.
    model.answer_model = AnswerVisualGeneratorSGLang(
        config["answerer_base_url"], config.get("answerer_api_key", "Qwen-7B"))
    return model, config


@torch.inference_mode()
def encode_dataset_features(retriever, image_loader, text_loader):
    device = retriever.device
    gallery = torch.empty((len(image_loader.dataset), 512), device=device)
    for indices, _, images in image_loader:
        gallery[indices.to(device)] = retriever.encode_image(images)

    texts = []
    for _, _, _, captions in text_loader:
        texts.extend(captions)
    query = []
    for start in range(0, len(texts), 256):
        query.append(retriever.encode_text(texts[start:start + 256]))
    query = torch.cat(query, dim=0)
    return F.normalize(query, dim=-1), F.normalize(gallery, dim=-1)


def metric_row(qfeats, gfeats, qids, gids, args):
    interactive_similarity = F.normalize(qfeats, dim=-1) @ F.normalize(gfeats, dim=-1).t()
    return metric_similarity(interactive_similarity, qids, gids)


def metric_similarity(similarity, qids, gids):
    from utils.metrics import cal_rank, per_sample_ranks

    t2i_cmc, t2i_mAP, t2i_mINP, _ = cal_rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    row = {
        "R1": float(t2i_cmc[0].item()),
        "R5": float(t2i_cmc[4].item()),
        "R10": float(t2i_cmc[9].item()),
        "mAP": float(t2i_mAP.item()),
        "mINP": float(t2i_mINP.item()),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--ravel-config", default=str(MAIN_ROOT / "config/ravel-main.yaml"))
    parser.add_argument("--output-dir", default=str(LLAVA_ROOT / "dataset/evaluation-records/irra-transfer"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--transfer-data-root", default=os.environ.get("RAVEL_TRANSFER_DATA_ROOT"))
    parser.add_argument("--irra-root", default=os.environ.get("RAVEL_IRRA_ROOT"))
    parser.add_argument("--smoke-batches", type=int, default=-1)
    parser.add_argument("--interact-round", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global RDE_DATA_ROOT, IRRA_ROOT
    if args.transfer_data_root:
        RDE_DATA_ROOT = Path(args.transfer_data_root).expanduser().resolve()
    if args.irra_root:
        IRRA_ROOT = Path(args.irra_root).expanduser().resolve()
    if not RDE_DATA_ROOT.exists():
        raise FileNotFoundError(f"Transfer dataset root not found: {RDE_DATA_ROOT}. Set --transfer-data-root or RAVEL_TRANSFER_DATA_ROOT.")
    if not IRRA_ROOT.exists():
        raise FileNotFoundError(f"IRRA project root not found: {IRRA_ROOT}. Set --irra-root or RAVEL_IRRA_ROOT.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda:0"

    specification, gallery, queries = load_benchmark(args.dataset)
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    image_loader = DataLoader(gallery, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    query_dataset = queries
    if args.smoke_batches > 0:
        query_dataset = LimitedQueryDataset(queries, args.smoke_batches * args.batch_size)
    text_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model, config = load_ravel(Path(args.ravel_config), args.dataset, args.interact_round)
    from engine_eval import _interactive_compute_embedding

    eval_args = Namespace(
        interact_round=args.interact_round,
        output_dir=str(output_dir),
        checkpoint_name=f"ravel_{args.dataset.lower().replace('-', '_')}_protocol_seed{args.seed}",
        distributed=False,
    )
    qfeats, gfeats, qids, gids = _interactive_compute_embedding(
        model, image_loader, text_loader, args.smoke_batches, eval_args)

    dataset_retriever = StandardIRRAAdapter(
        specification["irra_checkpoint"], specification["irra_config"], device)
    dataset_q, dataset_g = encode_dataset_features(dataset_retriever, image_loader, text_loader)
    if qfeats.shape[-1] != dataset_q.shape[-1] or gfeats.shape[-1] != dataset_g.shape[-1]:
        raise RuntimeError(
            f"Feature dimensions do not match: interactive={qfeats.shape[-1]}/{gfeats.shape[-1]}, "
            f"dataset={dataset_q.shape[-1]}/{dataset_g.shape[-1]}"
        )

    metric_args = Namespace(distributed=False)
    results = [metric_row(qfeats[r], gfeats, qids, gids, metric_args)
               | {"Round": r, "Protocol": "interactive_only"}
               for r in range(args.interact_round + 1)]
    interactive_similarity = F.normalize(qfeats[-1], dim=-1) @ F.normalize(gfeats, dim=-1).t()
    dataset_similarity = dataset_q @ dataset_g.t()
    fused_similarity = (interactive_similarity + dataset_similarity) / 2.0
    results.append(metric_similarity(fused_similarity, qids, gids)
                   | {"Round": args.interact_round, "Protocol": "similarity_average_after_five_rounds"})

    payload = {
        "dataset": args.dataset,
        "protocol": {
            "initial_query": "original_dataset_caption",
            "answerer": "multimodal_ground_truth_image",
            "questioner": str(config.get("question_model_path")),
            "interactive_retriever": str(LLAVA_ROOT / "weight/irra-interactive-pedes/checkpoint.pth"),
            "dataset_retriever": str(specification["irra_checkpoint"]),
            "similarity_fusion": "(S_interactive + S_dataset_specific) / 2 after round 5",
        },
        "results": results,
    }
    with (output_dir / "protocol_results.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    torch.save({
        "qfeats_interactive": qfeats.cpu(),
        "gfeats_interactive": gfeats.cpu(),
        "qfeats_dataset": dataset_q.cpu(),
        "gfeats_dataset": dataset_g.cpu(),
        "qids": qids.cpu(),
        "gids": gids.cpu(),
    }, output_dir / "protocol_features.pt")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
