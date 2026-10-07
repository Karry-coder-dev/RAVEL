#!/usr/bin/env python3
"""Compute the Section 3.1 question-quality diagnostics."""

import argparse
import json
import os
from pathlib import Path

from scipy.stats import pointbiserialr, spearmanr


def load_jsonl(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def association(rows):
    delta_rr = [
        1.0 / max(1, int(row["rank_after"]))
        - 1.0 / max(1, int(row["rank_before"]))
        for row in rows
    ]
    improvement = [int(row["rank_after"] < row["rank_before"]) for row in rows]
    labels = {
        "discriminative": [
            int((row.get("audit_json") or {}).get("discriminative_attribute") is True)
            for row in rows
        ],
        "useful": [
            int((row.get("audit_json") or {}).get("likely_useful_for_retrieval") is True)
            for row in rows
        ],
    }
    result = {
        "n": len(rows),
        "improvement_rate": sum(improvement) / len(rows),
        "gain_definition": "1 / rank_after - 1 / rank_before",
    }
    for name, values in labels.items():
        rho = spearmanr(values, delta_rr)
        r_pb = pointbiserialr(values, improvement)
        result[name] = {
            "positive_rate": sum(values) / len(values),
            "spearman_rho": rho.statistic,
            "spearman_p": rho.pvalue,
            "point_biserial_r": r_pb.statistic,
            "point_biserial_p": r_pb.pvalue,
        }
    return result


def kappa(human_rows, paired_rows):
    paired = {}
    for row in paired_rows:
        key = f"{row['group']}_{row['sample_index']}_{row['round']}"
        paired[key] = row
    complete = []
    for row in human_rows:
        usefulness = (row.get("annotation") or {}).get("usefulness")
        if usefulness and row["id"] in paired:
            complete.append((row, paired[row["id"]]))
    human_binary = [int(row["annotation"]["usefulness"] in {"partial", "useful"}) for row, _ in complete]
    human_score_map = {"useless": 0.0, "partial": 0.5, "useful": 1.0}
    human_scores = [
        human_score_map[row["annotation"]["usefulness"]]
        for row, _ in complete
    ]
    model = [
        int((row.get("audit_json") or {}).get("likely_useful_for_retrieval") is True)
        for _, row in complete
    ]
    human_rr_gain = [
        1.0 / max(1, int(row["rank_after"]))
        - 1.0 / max(1, int(row["rank_before"]))
        for row, _ in complete
    ]
    human_improvement = [
        int(row["rank_after"] < row["rank_before"])
        for row, _ in complete
    ]
    human_rho = spearmanr(human_scores, human_rr_gain)
    human_r_pb = pointbiserialr(human_binary, human_improvement)
    observed = sum(a == b for a, b in zip(human_binary, model)) / len(complete)
    p_human = sum(human_binary) / len(complete)
    p_model = sum(model) / len(complete)
    expected = p_human * p_model + (1 - p_human) * (1 - p_model)
    return {
        "pack_size": len(human_rows),
        "completed_pairs": len(complete),
        "human_positive_rate": p_human,
        "model_positive_rate": p_model,
        "observed_agreement": observed,
        "cohens_kappa": (observed - expected) / (1 - expected),
        "usefulness_vs_reciprocal_rank_gain": {
            "spearman_rho": human_rho.statistic,
            "spearman_p": human_rho.pvalue,
            "point_biserial_r": human_r_pb.statistic,
            "point_biserial_p": human_r_pb.pvalue,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-glob",
        default="analysis_outputs/qwen3vl32_quality_20260914/qwen3vl32_quality_part*.jsonl",
    )
    parser.add_argument(
        "--human",
        default=os.environ.get("RAVEL_HUMAN_AUDIT_JSON"),
    )
    parser.add_argument(
        "--paired-model",
        default="manual_question_audit_pack/qwen3vl32_question_quality_manual200.jsonl",
    )
    parser.add_argument(
        "--out",
        default="analysis_outputs/qwen3vl32_quality_20260914/metrics_summary.json",
    )
    args = parser.parse_args()
    if not args.human:
        parser.error("provide --human or set RAVEL_HUMAN_AUDIT_JSON")

    model_rows = load_jsonl(sorted(Path().glob(args.model_glob)))
    human_rows = json.loads(Path(args.human).read_text(encoding="utf-8"))["rows"]
    paired_rows = load_jsonl([args.paired_model])
    summary = {
        "overall": association(model_rows),
        "per_round": {
            str(round_id): association(
                [row for row in model_rows if row["round"] == round_id]
            )
            for round_id in sorted({row["round"] for row in model_rows})
        },
        "human_model": kappa(human_rows, paired_rows),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
