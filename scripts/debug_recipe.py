#!/usr/bin/env python3
"""Debug RECIPE injection: retrieval correctness + hook effectiveness.

For N test queries:
  1. Show which edit was retrieved (top-k) vs the ground-truth matching edit
  2. Generate WITH edits active
  3. Generate WITH edits cleared (baseline)
  4. Compare outputs — if identical, the hook has no effect

Usage:
  python scripts/debug_recipe.py \
    --checkpoint external/RECIPE/.../epoch-159-i-99000-ema_loss-0.2240 \
    --edits data/edit_pairs.json \
    --n_queries 8 \
    --quantization int4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.recipe_official import RECIPEOfficialInference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--edits", required=True)
    p.add_argument("--n_queries", type=int, default=8)
    p.add_argument("--quantization", default="int4", choices=["int4", "int8", "bf16"])
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.edits) as f:
        edits = json.load(f)
    print(f"Loaded {len(edits)} edits from {args.edits}")

    pipe = RECIPEOfficialInference(
        checkpoint_path=args.checkpoint,
        quantization=args.quantization,
        max_new_tokens=30,
        do_sample=False,
    )
    pipe.apply_edits(edits)
    editor = pipe._editor
    tokenizer = pipe._tokenizer
    model = editor.model

    print(f"\nRECIPE state:")
    print(f"  retr_top_k    = {editor.retr_top_k}")
    print(f"  retr_min_sim  = {editor.retr_min_sim}")
    print(f"  auto_retrieve = {editor.auto_retrieve}")
    print(f"  prompts_base  = {editor.prompts_base.shape} "
          f"(nonzero rows: {(editor.prompts_base.abs().sum(dim=(1,2)) > 0).sum().item()})")
    print(f"  knowledge_base       shape = {editor.knowledge_base.shape}")
    print(f"  knowledge_base_nl    len   = {len(editor.knowledge_base_nl)}")

    # Hook plumbing diagnostics
    print(f"\nHook plumbing:")
    print(f"  model type               = {type(model).__name__}")
    print(f"  model.forward type       = {type(model.forward).__name__}")
    print(f"  model.forward.__name__   = {getattr(model.forward, '__name__', 'N/A')}")
    print(f"  model has 'recipe_hooked'= {hasattr(model, 'recipe_hooked')}")
    print(f"  begin_layer type         = {type(editor.begin_layer).__name__}")
    print(f"  begin_layer._forward_pre_hooks = {len(editor.begin_layer._forward_pre_hooks)}")
    print(f"  lm_head._forward_hooks       = {len(editor.lm_head._forward_hooks)}")

    # Wrap forward_recipe to count calls
    call_counter = {"n": 0, "last_past_kv": None, "last_input_shape": None}
    orig_forward = model.forward
    def counting_forward(**kargs):
        call_counter["n"] += 1
        call_counter["last_past_kv"] = kargs.get("past_key_values") is not None
        ii = kargs.get("input_ids")
        call_counter["last_input_shape"] = tuple(ii.shape) if ii is not None else None
        return orig_forward(**kargs)
    model.forward = counting_forward
    print(f"  Wrapped counting_forward over original forward.")

    # Sample from each split
    test_queries = edits[:args.n_queries]

    for i, edit in enumerate(test_queries):
        query = edit["question"]
        gold = edit["answer"]
        print("\n" + "=" * 80)
        print(f"[{i}] QUERY : {query}")
        print(f"    GOLD  : {gold}")

        # 1. Retrieval: top-3 similarities
        with torch.no_grad():
            retrieved_ids, (sorted_sim, order) = editor.retrieve_and_get_ids_sim([query])
            top3_ids = order[0, :3].tolist()
            top3_sims = sorted_sim[0, :3].tolist()
            prot_sim = sorted_sim[0, 0].item() if order[0, 0].item() == 0 else None

        print(f"    TOP-3 retrieval:")
        for j, (eid, sim) in enumerate(zip(top3_ids, top3_sims)):
            text = editor.knowledge_base_nl[eid]
            marker = " <-- PROTOTYPE" if eid == 0 else ""
            print(f"      {j}. sim={sim:.4f} id={eid} text={text[:80]!r}{marker}")

        # Check if the TRUE matching edit (this same question) is in top-1
        # editor.knowledge_base_nl[0] is prototype; edits are at indices 1..N
        # Our edit list in Python `edits[i]` corresponds to knowledge_base_nl[i+1]
        true_edit_id = i + 1
        true_rank = (order[0] == true_edit_id).nonzero(as_tuple=True)[0].item()
        print(f"    TRUE edit id = {true_edit_id} (rank {true_rank} in retrieval)")

        # Check prompts_base for retrieved id
        top1_id = retrieved_ids[0]
        if len(top1_id) > 0:
            pid = top1_id[0].item()
            prompt_tensor = editor.prompts_base[pid]
            print(f"    INJECTED prompt (id={pid}): shape={prompt_tensor.shape} "
                  f"norm={prompt_tensor.norm().item():.4f}")

        # 2. Generate WITH edits active
        tok = tokenizer(query, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = tok["input_ids"].to(model.device)
        attn = tok["attention_mask"].to(model.device)

        # Reset adopted_prompts so we can verify forward_recipe wrote to it
        editor.adopted_prompts = []
        call_counter["n"] = 0
        with torch.no_grad():
            out_with = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=15, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        resp_with = tokenizer.decode(out_with[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        n_calls_with = call_counter["n"]
        ap_after_with = len(editor.adopted_prompts)
        has_past_kv_attr = hasattr(editor.begin_layer, "has_past_kv")
        print(f"    [with edits] forward calls={n_calls_with} "
              f"adopted_prompts_after={ap_after_with} "
              f"begin_layer.has_past_kv_attr={has_past_kv_attr}")

        # 3. Generate WITHOUT edits for A/B comparison
        editor.auto_retrieve = False
        editor.adopted_prompts = [torch.zeros([0, editor.cfg.model_hidden_size],
                                              device=editor.device,
                                              dtype=editor.prompts_base.dtype)]
        with torch.no_grad():
            out_wo = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=15, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        resp_wo = tokenizer.decode(out_wo[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        editor.auto_retrieve = True

        print(f"    GEN with edits : {resp_with!r}")
        print(f"    GEN no   edits : {resp_wo!r}")
        print(f"    IDENTICAL?     : {resp_with == resp_wo}")
        print(f"    Contains gold? : with={gold.lower() in resp_with.lower()}, "
              f"without={gold.lower() in resp_wo.lower()}")

    print("\n" + "=" * 80)
    print("Debug complete.")


if __name__ == "__main__":
    main()
