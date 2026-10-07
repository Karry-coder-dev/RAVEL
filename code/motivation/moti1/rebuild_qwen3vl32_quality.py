"""Rebuild the sampled interaction states used by the question-quality audit."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch


def rank_positions(qfeats, gfeats, qids, gids):
    ranks = []
    gids = gids.long()
    for round_feats in qfeats:
        order = torch.argsort(round_feats @ gfeats.t(), dim=1, descending=True)
        same = gids[order] == qids.long().unsqueeze(1)
        ranks.append(same.float().argmax(dim=1).long() + 1)
    return torch.stack(ranks, dim=0).cpu()


def build_records(conv_log, max_rounds):
    conv = torch.load(conv_log, map_location="cpu")
    logs = conv["conversation_log"]
    ranks = rank_positions(
        conv["qfeats"].float(),
        conv["gfeats"].float(),
        conv["qids"].long(),
        conv["gids"].long(),
    )
    records = []
    for sample_index, log in enumerate(logs):
        history_questions = []
        for round_index, step in enumerate(log["interaction"][:max_rounds]):
            rank_before = int(ranks[round_index, sample_index])
            rank_after = int(ranks[round_index + 1, sample_index])
            rank_delta = rank_before - rank_after
            group = "up" if rank_delta > 0 else "down" if rank_delta < 0 else "same"
            records.append(
                {
                    "sample_index": sample_index,
                    "round": round_index,
                    "group": group,
                    "rank_before": rank_before,
                    "rank_after": rank_after,
                    "rank_delta": rank_delta,
                    "initial_query": log.get("initial_query", ""),
                    "history_questions": history_questions[:],
                    "question": step.get("question", ""),
                    "answer": step.get("answer", ""),
                    "candidates": step.get("candidate", [])[:4],
                }
            )
            history_questions.append(step.get("question", ""))
    return records


def sample_records(records, quotas, max_rounds, seed):
    rng = random.Random(seed)
    selected = []
    for group, quota in quotas.items():
        if quota <= 0:
            continue
        bucket = [record for record in records if record["group"] == group]
        if len(bucket) < quota:
            raise RuntimeError(f"Not enough {group} states: requested {quota}, found {len(bucket)}")
        by_round = defaultdict(list)
        for record in bucket:
            by_round[record["round"]].append(record)
        group_selected = []
        per_round = quota // max_rounds
        for round_index in range(max_rounds):
            items = by_round.get(round_index, [])[:]
            rng.shuffle(items)
            group_selected.extend(items[:per_round])
        used = {(r["sample_index"], r["round"]) for r in group_selected}
        rest = [r for r in bucket if (r["sample_index"], r["round"]) not in used]
        rng.shuffle(rest)
        group_selected.extend(rest[: quota - len(group_selected)])
        selected.extend(group_selected[:quota])
    rng.shuffle(selected)
    for index, record in enumerate(selected):
        record["id"] = f"{record['group']}_{record['sample_index']}_{record['round']}"
        record["audit_index"] = index
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--num-down", type=int, default=1500)
    parser.add_argument("--num-up", type=int, default=500)
    parser.add_argument("--num-same", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    records = build_records(args.conv_log, args.max_rounds)
    selected = sample_records(
        records,
        {"down": args.num_down, "up": args.num_up, "same": args.num_same},
        args.max_rounds,
        args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "source_conv_log": str(args.conv_log),
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "requested_counts": {"down": args.num_down, "up": args.num_up, "same": args.num_same},
        "total_records": len(selected),
        "actual_counts": dict(Counter(r["group"] for r in selected)),
    }
    Path(args.meta).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
