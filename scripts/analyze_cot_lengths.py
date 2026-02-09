#!/usr/bin/env python3
"""Analyze token lengths with CoT (<think> blocks) preserved.

Measures actual tokenized sequence lengths using the training-safe template
that preserves <think> blocks, to determine the right max_seq_length for
training without OOM.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
import numpy as np

# Training-safe template (same as in trainer.py)
TRAINING_CHAT_TEMPLATE = (
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
    data_path = "src/data/dataset_final.json"
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE

    print(f"Loading data: {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # RAG user prefix (matches training)
    user_prefix = (
        "Answer the question based ONLY on the provided documents. "
        "If the answer is not in the documents, say so clearly.\n\n"
    )

    lengths = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        analysis = item.get("analysis", "")
        evidence = item.get("evidence_snippet", "")

        # RAG format
        context = ""
        if evidence:
            context = f"[Documents:]\n--- Document 1 ---\n{evidence}\n\n"
        user_content = f"{user_prefix}{context}[Question:]\n{question}"

        # CoT format
        if analysis:
            assistant_content = f"<think>\n{analysis}\n</think>\n\n{answer}"
        else:
            assistant_content = answer

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    lengths = np.array(lengths)

    print(f"\n{'='*60}")
    print("TOKEN LENGTH DISTRIBUTION (with <think> blocks preserved)")
    print(f"{'='*60}")
    print(f"  Count:  {len(lengths)}")
    print(f"  Min:    {lengths.min()}")
    print(f"  Mean:   {lengths.mean():.0f}")
    print(f"  Median: {np.median(lengths):.0f}")
    print(f"  P75:    {np.percentile(lengths, 75):.0f}")
    print(f"  P90:    {np.percentile(lengths, 90):.0f}")
    print(f"  P95:    {np.percentile(lengths, 95):.0f}")
    print(f"  P99:    {np.percentile(lengths, 99):.0f}")
    print(f"  Max:    {lengths.max()}")

    for cutoff in [1024, 1536, 2048, 2560, 3072, 3584, 4096]:
        pct = (lengths <= cutoff).sum() / len(lengths) * 100
        truncated = (lengths > cutoff).sum()
        print(f"\n  max_seq_length={cutoff}: {pct:.1f}% fit, {truncated} truncated")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
