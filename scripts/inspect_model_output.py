#!/usr/bin/env python3
"""
Inspect raw model output from llama.cpp server.

Sends test prompts in multiple formats and shows the full raw response,
highlighting whether the model properly uses <think> tags and generates
content after them.

Usage:
    python scripts/inspect_model_output.py
    python scripts/inspect_model_output.py --url http://localhost:8080
    python scripts/inspect_model_output.py --prompt "Was ist ein QM-Audit?"
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


# Training prompt formats (must match src/data_loaders/local_loader.py)
SIMPLE_PREFIX = "Answer the following question accurately and concisely based on your knowledge.\n\n"
RAG_PREFIX = (
    "Answer the question based ONLY on the provided documents. "
    "If the answer is not in the documents, say so clearly.\n\n"
)


def query_model(url: str, messages: list[dict], temperature: float = 0.6, max_tokens: int = 4096) -> dict:
    """Send a chat completion request to the llama.cpp server."""
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Error connecting to {endpoint}: {e}")
        print("Is the llama.cpp server running?")
        sys.exit(1)


def analyze_response(content: str, finish_reason: str):
    """Analyze the model response for think tag structure."""
    print(content)
    print("=" * 70)
    print()

    has_open = "<think>" in content
    has_close = "</think>" in content

    print(f"  Finish reason:  {finish_reason}")
    print(f"  Total length:   {len(content)} chars")
    print(f"  Has <think>:    {has_open}")
    print(f"  Has </think>:   {has_close}")

    if has_open and has_close:
        think_start = content.index("<think>")
        think_end = content.index("</think>") + len("</think>")
        think_content = content[think_start + len("<think>"):think_end - len("</think>")]
        after_think = content[think_end:].strip()

        print(f"  Think block:    {len(think_content)} chars")
        print(f"  After </think>: {len(after_think)} chars")

        if after_think:
            print(f"  RESULT: OK - answer present after </think>")
        else:
            print(f"  RESULT: PROBLEM - no content after </think>!")

    elif has_open and not has_close:
        print(f"  RESULT: PROBLEM - <think> opened but never closed!")

    elif not has_open:
        print(f"  RESULT: No <think> tags - model not using CoT")

    if finish_reason == "length":
        print(f"  WARNING: finish_reason='length' - ran out of tokens!")

    print()


def run_test(url: str, label: str, messages: list[dict], temperature: float, max_tokens: int):
    """Run a single test and analyze the result."""
    print(f"{'=' * 70}")
    print(f"TEST: {label}")
    print(f"{'=' * 70}")
    print(f"User message:")
    user_msg = messages[-1]["content"]
    # Show first 200 chars of user message
    if len(user_msg) > 200:
        print(f"  {user_msg[:200]}...")
    else:
        print(f"  {user_msg}")
    print()
    print("Model output:")
    print("-" * 70)

    response = query_model(url, messages, temperature, max_tokens)
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content", "") or ""
    finish_reason = choice.get("finish_reason", "unknown")
    tokens = response.get("usage", {})

    # Check for reasoning/thinking in separate fields
    reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or ""

    if reasoning:
        print(f"[REASONING FIELD DETECTED - {len(reasoning)} chars]:")
        print(reasoning[:500])
        if len(reasoning) > 500:
            print(f"  ... ({len(reasoning)} chars total)")
        print()
        print(f"[CONTENT FIELD - {len(content)} chars]:")

    analyze_response(content, finish_reason)

    print(f"  Tokens - prompt: {tokens.get('prompt_tokens', '?')}, "
          f"completion: {tokens.get('completion_tokens', '?')}, "
          f"total: {tokens.get('total_tokens', '?')}")

    # Dump full message object to see ALL fields
    print(f"  Full message keys: {list(message.keys())}")
    # Show any non-standard fields
    for key in message:
        if key not in ("role", "content"):
            val = str(message[key])
            print(f"  message['{key}']: {val[:200]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect raw model output from llama.cpp")
    parser.add_argument("--url", default="http://localhost:8080", help="llama.cpp server URL")
    parser.add_argument("--prompt", default="Welche Anforderungen gelten für die Akkreditierung?", help="Test prompt")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Server:      {args.url}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens:  {args.max_tokens}")
    print()

    # Test 1: Bare prompt (what OpenWebUI likely sends)
    run_test(
        args.url,
        "BARE PROMPT (no training format)",
        [{"role": "user", "content": args.prompt}],
        args.temperature,
        args.max_tokens,
    )

    # Test 2: Simple training format (monolithic)
    run_test(
        args.url,
        "SIMPLE FORMAT (matches monolithic training)",
        [{"role": "user", "content": f"{SIMPLE_PREFIX}[Question:]\n{args.prompt}"}],
        args.temperature,
        args.max_tokens,
    )

    # Test 3: RAG training format (with placeholder context)
    rag_prompt = (
        f"{RAG_PREFIX}"
        f"[Documents:]\n--- Document 1 ---\n"
        f"Die Akkreditierung erfolgt gemäß den Anforderungen der EN ISO/IEC 17025.\n\n"
        f"[Question:]\n{args.prompt}"
    )
    run_test(
        args.url,
        "RAG FORMAT (matches RAG training)",
        [{"role": "user", "content": rag_prompt}],
        args.temperature,
        args.max_tokens,
    )

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If SIMPLE/RAG formats produce <think>...</think> + answer,")
    print("but BARE does not, then OpenWebUI needs to prepend the")
    print("training prefix to user messages (via system prompt or")
    print("model prompt template in OpenWebUI settings).")
    print()
    print("If NONE produce <think> tags, the issue is likely the")
    print("chat template (check --jinja flag) or the fine-tuning")
    print("didn't persist through GGUF conversion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
