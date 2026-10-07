import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from model.llava_qwen_reid import LlavaQwenForPersonReID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(args.out)

    print(f"Loading base: {args.base}")
    model = LlavaQwenForPersonReID.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    print("Merging adapter...")
    model = model.merge_and_unload()
    print(f"Saving merged model: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
