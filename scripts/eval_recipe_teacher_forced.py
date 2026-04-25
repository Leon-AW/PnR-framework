#!/usr/bin/env python3
"""Teacher-forced Edit-Success evaluation for RECIPE.

Reproduces the paper's metric (external/RECIPE/evaluation/editor_eval.py::
accuracy_and_prediction): a single forward pass over concat(question, target),
argmax per position, accuracy on the target token positions. No autoregressive
decoding — gold previous tokens are provided as context at every step.

Use alongside (not instead of) the free-form EM from eval_pnr.py. The gap
between the two reveals how much RECIPE's effect relies on teacher forcing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.recipe_official import RECIPEOfficialInference
from src.eval.dataset import build_situated_qa_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--edits", required=True)
    p.add_argument("--eval_sets", nargs="+",
                   default=["base", "temporal", "geo_india", "geo_australia"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--quantization", default="int4",
                   choices=["int4", "int8", "bf16"])
    p.add_argument("--output", default="eval_results/recipe_tf_es.json")
    return p.parse_args()


@torch.no_grad()
def teacher_forced_accuracy(editor, tokenizer, question: str, answer: str):
    """Return (token_acc, first_token_correct).

    token_acc: fraction of target tokens where argmax next-token logit equals gold.
    first_token_correct: 1.0 if the very first target token is predicted correctly.
    """
    sep = "" if answer.startswith((" ", "\t", "\n")) else " "
    ids_q = tokenizer(question, return_tensors="pt").input_ids[0]
    ids_full = tokenizer(question + sep + answer, return_tensors="pt").input_ids[0]
    device = editor.model.device
    ids_q = ids_q.to(device)
    ids_full = ids_full.to(device)

    n_q = ids_q.shape[0]
    n_tgt = ids_full.shape[0] - n_q
    if n_tgt <= 0:
        return float("nan"), float("nan")

    logits = editor.model(input_ids=ids_full.unsqueeze(0)).logits[0]
    pred_next = logits[:-1].argmax(-1)     # position i predicts token i+1
    gold_next = ids_full[1:]

    tgt_slice = slice(n_q - 1, n_q - 1 + n_tgt)
    correct = (pred_next[tgt_slice] == gold_next[tgt_slice])
    token_acc = correct.float().mean().item()
    first_correct = correct[0].float().item()
    return token_acc, first_correct


def main():
    args = parse_args()
    with open(args.edits) as f:
        edits = json.load(f)
    print(f"Loaded {len(edits)} edits")

    pipe = RECIPEOfficialInference(
        checkpoint_path=args.checkpoint,
        quantization=args.quantization,
        max_new_tokens=8,
        do_sample=False,
    )
    pipe.apply_edits(edits)
    editor = pipe._editor
    tokenizer = pipe._tokenizer

    report = {"by_split": {}, "config": vars(args)}

    all_tok, all_first = [], []
    for split in args.eval_sets:
        print(f"\n=== {split} ===")
        samples = build_situated_qa_dataset(split=split, n_samples=args.n_samples)

        tok_accs, first_accs = [], []
        for i, s in enumerate(samples):
            gold = s.gold_answers[0]
            ta, fa = teacher_forced_accuracy(editor, tokenizer, s.question, gold)
            if ta == ta:  # not nan
                tok_accs.append(ta)
                first_accs.append(fa)
            if i < 3:
                print(f"  [{i}] q={s.question[:50]!r} gold={gold!r} "
                      f"tok_acc={ta:.3f} first={fa:.0f}")

        mean_tok = sum(tok_accs) / max(1, len(tok_accs))
        mean_first = sum(first_accs) / max(1, len(first_accs))
        report["by_split"][split] = {
            "n": len(tok_accs),
            "tf_token_acc": mean_tok,
            "tf_first_token_acc": mean_first,
        }
        print(f"  → n={len(tok_accs)} tf_token_acc={mean_tok:.4f} "
              f"tf_first={mean_first:.4f}")
        all_tok.extend(tok_accs)
        all_first.extend(first_accs)

    report["summary"] = {
        "n": len(all_tok),
        "tf_token_acc": sum(all_tok) / max(1, len(all_tok)),
        "tf_first_token_acc": sum(all_first) / max(1, len(all_first)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
