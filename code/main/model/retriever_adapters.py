import importlib.util
import math
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class RDERetrieverAdapter(nn.Module):
    """Expose RDE BGE+TSE features through RAVEL's retriever interface."""

    def __init__(self, rde_model):
        super().__init__()
        self.rde_model = rde_model.eval()

    @staticmethod
    def _combine(base, tse):
        base = F.normalize(base.float(), dim=-1)
        tse = F.normalize(tse.float(), dim=-1)
        return torch.cat((base, tse), dim=-1) / math.sqrt(2.0)

    @torch.inference_mode()
    def encode_image(self, images):
        base, tse = self.rde_model.encode_image_full(images)
        return self._combine(base, tse).unsqueeze(1)

    @torch.inference_mode()
    def encode_text(self, tokens):
        base, tse = self.rde_model.encode_text_full(tokens)
        return self._combine(base, tse)


def load_rde_retriever(project_root, checkpoint_dir):
    project_root = Path(project_root).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_path = checkpoint_dir / "best.pth"
    config_path = checkpoint_dir / "configs.yaml"
    if not (project_root / "model").exists():
        raise FileNotFoundError(f"RDE project model directory not found: {project_root / 'model'}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RDE checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"RDE config not found: {config_path}")

    with config_path.open() as handle:
        rde_args = Namespace(**yaml.load(handle, Loader=yaml.FullLoader))
    rde_args.training = False

    model_dir = project_root / "model"
    spec = importlib.util.spec_from_file_location(
        "ravel_rde_model",
        model_dir / "__init__.py",
        submodule_search_locations=[str(model_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rde = module.build_model(rde_args, num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    rde.load_state_dict(checkpoint["model"], strict=True)
    return RDERetrieverAdapter(rde.cuda()), int(rde_args.text_length)
