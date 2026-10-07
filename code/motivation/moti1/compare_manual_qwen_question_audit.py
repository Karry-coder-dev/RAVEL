import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROBLEM_MAP = {
    "shared_attribute": "shared_attribute",
    "absent_or_invisible": "absent_or_invisible_attribute",
    "generic_background": "generic_background",
    "negative_unknown_answer": "negative_or_unknown_answer",
    "redundant": "redundant_with_history",
    "image_too_blurry": "uncertain_visual_evidence",
}


def load_model(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['group']}_{row['sample_index']}_{row['round']}"
            rows[key] = row
    return rows


def load_human(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for row in data["rows"]:
        ann = row.get("annotation") or {}
        if not ann.get("usefulness"):
            continue
        rows[row["id"]] = row
    return rows


def pct(num, den):
    return round(100.0 * num / den, 2) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="manual_question_audit_annotations.json exported from the web page")
    ap.add_argument("--model", default="manual_question_audit_pack/qwen3vl32_question_quality_manual200.jsonl")
    ap.add_argument("--out", default="manual_question_audit_pack/manual_vs_qwen3vl32_comparison.csv")
    args = ap.parse_args()

    human = load_human(args.human)
    model = load_model(args.model)
    common = sorted(set(human) & set(model))

    rows = []
    useful_agree = 0
    group_stats = defaultdict(lambda: Counter())
    problem_stats = {k: Counter() for k in PROBLEM_MAP}

    for key in common:
        hrow = human[key]
        mrow = model[key]
        ann = hrow["annotation"]
        mj = mrow.get("audit_json") or {}
        human_useful = ann.get("usefulness") in {"useful", "partial"}
        model_useful = mj.get("likely_useful_for_retrieval") is True
        useful_agree += int(human_useful == model_useful)
        group = hrow["group"]
        group_stats[group]["n"] += 1
        group_stats[group]["human_useful"] += int(human_useful)
        group_stats[group]["model_useful"] += int(model_useful)
        group_stats[group]["useful_agree"] += int(human_useful == model_useful)
        for hkey, mkey in PROBLEM_MAP.items():
            hv = ann.get(hkey) is True
            mv = mj.get(mkey) is True
            problem_stats[hkey]["n"] += 1
            problem_stats[hkey]["human_true"] += int(hv)
            problem_stats[hkey]["model_true"] += int(mv)
            problem_stats[hkey]["agree"] += int(hv == mv)
        rows.append({
            "id": key,
            "group": group,
            "round": hrow["round"],
            "rank_before": hrow["rank_before"],
            "rank_after": hrow["rank_after"],
            "rank_delta": hrow["rank_delta"],
            "question": hrow["question"],
            "answer": hrow["answer"],
            "human_usefulness": ann.get("usefulness", ""),
            "model_likely_useful": model_useful,
            "model_reason": mj.get("reason", ""),
            "human_note": ann.get("note", ""),
        })

    print(f"matched labeled rows: {len(common)}")
    print(f"useful agreement: {useful_agree}/{len(common)} ({pct(useful_agree, len(common))}%)")
    print("\nBy group:")
    for group, c in sorted(group_stats.items()):
        print(
            f"{group}: n={c['n']} "
            f"human_useful={pct(c['human_useful'], c['n'])}% "
            f"model_useful={pct(c['model_useful'], c['n'])}% "
            f"agree={pct(c['useful_agree'], c['n'])}%"
        )
    print("\nProblem label agreement:")
    for key, c in problem_stats.items():
        print(
            f"{key}: human={pct(c['human_true'], c['n'])}% "
            f"model={pct(c['model_true'], c['n'])}% "
            f"agree={pct(c['agree'], c['n'])}%"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
