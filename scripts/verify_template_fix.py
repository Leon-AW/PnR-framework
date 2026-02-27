#!/usr/bin/env python3
"""Verify that the training-safe chat template preserves <think> blocks.

Compares the STOCK DeepSeek-R1 template (which strips <think>) against
our training-safe template (which preserves it). This confirms the fix
that was the root cause of the model not generating thinking tokens.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

# Training-safe template (same as in trainer.py)
TRAINING_SAFE_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "<\uff5cUser\uff5c>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "<\uff5cAssistant\uff5c>{{ message['content'] }}<\uff5cend\u2581of\u2581sentence\uff5c>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<\uff5cAssistant\uff5c>{% endif %}"
)


def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    data_path = "src/data/dataset_final.json"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load one sample
    with open(data_path) as f:
        data = json.load(f)

    item = data[0]
    question = item["question"]
    answer = item["answer"]
    analysis = item.get("analysis", "")

    # Build messages (same as local_loader.py RAG format)
    user_prefix = (
        "Answer the question based ONLY on the provided documents. "
        "If the answer is not in the documents, say so clearly.\n\n"
    )
    evidence = item.get("evidence_snippet", "")
    context = f"[Documents:]\n--- Document 1 ---\n{evidence}\n\n" if evidence else ""
    user_content = f"{user_prefix}{context}[Question:]\n{question}"

    # CoT format
    assistant_content = f"<think>\n{analysis}\n</think>\n\n{answer}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # =========================================================================
    # Test 1: STOCK template (the buggy one)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: STOCK DeepSeek-R1 template (BUGGY - strips <think>)")
    print("=" * 70)
    # Use the original tokenizer template (not overridden)
    stock_result = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    has_think_stock = "<think>" in stock_result
    print(f"Contains <think>: {has_think_stock}")
    # Show the assistant part only
    if "<｜Assistant｜>" in stock_result:
        assistant_part = stock_result.split("<｜Assistant｜>")[-1]
        print(f"Assistant output (first 300 chars):\n{assistant_part[:300]}")

    # =========================================================================
    # Test 2: TRAINING-SAFE template (our fix)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: TRAINING-SAFE template (FIXED - preserves <think>)")
    print("=" * 70)
    tokenizer.chat_template = TRAINING_SAFE_TEMPLATE
    safe_result = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    has_think_safe = "<think>" in safe_result
    print(f"Contains <think>: {has_think_safe}")
    if "<｜Assistant｜>" in safe_result:
        assistant_part = safe_result.split("<｜Assistant｜>")[-1]
        print(f"Assistant output (first 300 chars):\n{assistant_part[:300]}")

    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if not has_think_stock and has_think_safe:
        print("CONFIRMED: Stock template STRIPS <think>, training-safe template PRESERVES it.")
        print("The model WILL learn to generate <think> blocks with the fixed template.")
    elif has_think_stock and has_think_safe:
        print("Both templates preserve <think> (unexpected).")
    elif not has_think_safe:
        print("ERROR: Training-safe template also strips <think>! Fix needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
