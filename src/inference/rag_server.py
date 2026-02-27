"""
FastAPI RAG Server
==================

OpenAI-compatible API server that sits between OpenWebUI and llama.cpp,
providing sophisticated RAG retrieval with hybrid search and reranking.

Endpoints:
- GET  /v1/models          — Model discovery for OpenWebUI
- POST /v1/chat/completions — Main chat endpoint (streaming + non-streaming)
- GET  /health              — Health check
- POST /admin/reindex       — Reindex data sources
- GET  /admin/stats         — Pipeline statistics

Entry point:
    python -m src.inference.rag_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import traceback
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.inference.query_pipeline import QueryPipeline
from src.inference.rag_config import RAGServerConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Globals
# =============================================================================

config: RAGServerConfig = None  # type: ignore
pipeline: QueryPipeline = None  # type: ignore
http_client: httpx.AsyncClient = None  # type: ignore


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline and HTTP client on startup."""
    global config, pipeline, http_client

    config = RAGServerConfig.from_env()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting RAG server...")
    logger.info(f"llama.cpp backend: {config.llama_url}")

    # Load pipeline
    pipeline = QueryPipeline(config)
    pipeline.load()

    # Create async HTTP client for llama.cpp proxy
    http_client = httpx.AsyncClient(
        base_url=config.llama_url,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )

    logger.info(
        f"RAG server ready on {config.host}:{config.port} — "
        f"sources: {pipeline.loaded_sources}"
    )

    yield

    # Cleanup
    await http_client.aclose()
    logger.info("RAG server shut down")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Advanced RAG Server",
    description="OpenAI-compatible RAG proxy for QM documents",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Model Discovery
# =============================================================================

@app.get("/v1/models")
async def list_models():
    """Return available models (proxied from llama.cpp + metadata)."""
    try:
        resp = await http_client.get("/v1/models")
        data = resp.json()
    except Exception:
        # Fallback if llama.cpp is unreachable
        data = {
            "object": "list",
            "data": [
                {
                    "id": "rag-qm-model",
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }
    return JSONResponse(content=data)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check: pipeline status + llama.cpp connectivity."""
    llama_ok = False
    try:
        resp = await http_client.get("/health")
        llama_ok = resp.status_code == 200
    except Exception:
        pass

    return JSONResponse(content={
        "status": "ok" if pipeline.is_loaded and llama_ok else "degraded",
        "pipeline_loaded": pipeline.is_loaded,
        "llama_cpp_ok": llama_ok,
        "loaded_sources": pipeline.loaded_sources,
    })


# =============================================================================
# Chat Completions
# =============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main chat endpoint — runs RAG pipeline then proxies to llama.cpp."""
    try:
        body = await request.json()

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Extract user message and history
        user_message = ""
        history = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue  # We build our own system prompt
            if role in ("user", "assistant"):
                history.append({"role": role, "content": content})

        if history and history[-1]["role"] == "user":
            user_message = history[-1]["content"]
            history = history[:-1]  # Remove last user msg from history
        else:
            user_message = messages[-1].get("content", "")

        # Run RAG pipeline in a thread to avoid blocking the event loop
        # (embedding, FAISS search, reranking are all CPU/GPU-bound)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(pipeline.run, user_message, history)
        )

        # Build llama.cpp request
        llama_body = {
            "messages": result.messages,
            "model": body.get("model", "default"),
            "stream": stream,
            "temperature": body.get("temperature", config.default_temperature),
            "max_tokens": body.get("max_tokens", config.default_max_tokens),
            "frequency_penalty": body.get("frequency_penalty", config.default_frequency_penalty),
            "presence_penalty": body.get("presence_penalty", config.default_presence_penalty),
        }

        # Pass through optional params
        for key in ("top_p", "stop"):
            if key in body:
                llama_body[key] = body[key]

        language = result.query_analysis.language
        hyde_text = result.metadata.get("hyde_document", "")

        if stream:
            return StreamingResponse(
                _stream_response(llama_body, result.citations, language, hyde_text),
                media_type="text/event-stream",
            )
        else:
            return await _non_stream_response(llama_body, result.citations, language, hyde_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completions error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Streaming Response
# =============================================================================

async def _stream_response(
    llama_body: dict, citations: list, language: str = "de", hyde_text: str = ""
) -> AsyncGenerator[str, None]:
    """Stream SSE response from llama.cpp with think-token stripping and citation footer.

    Buffers up to 7 characters to detect <think> opening/closing tags
    at chunk boundaries.
    """
    try:
        async with http_client.stream(
            "POST",
            "/v1/chat/completions",
            json=llama_body,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                yield f"data: {json.dumps({'error': error_body.decode()})}\n\n"
                yield "data: [DONE]\n\n"
                return

            buffer = ""
            in_think = False

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    # Flush any remaining buffer
                    if buffer and not in_think and config.enable_think_stripping:
                        yield _make_sse_chunk(buffer)
                        buffer = ""

                    # Append HyDE expansion section
                    if config.enable_hyde_visibility and hyde_text:
                        hyde_section = _format_hyde_section(hyde_text, language)
                        yield _make_sse_chunk(hyde_section)

                    # Append citation footer
                    if config.enable_citations and citations:
                        footer = _format_citation_footer(citations, language)
                        yield _make_sse_chunk(footer)

                    yield "data: [DONE]\n\n"
                    return

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract content delta
                delta_content = ""
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    delta_content = delta.get("content", "")

                if not delta_content:
                    # Forward non-content chunks as-is
                    yield f"data: {data_str}\n\n"
                    continue

                if not config.enable_think_stripping:
                    yield f"data: {data_str}\n\n"
                    continue

                # Buffer-based think token stripping
                buffer += delta_content

                # Process buffer
                output = ""
                while buffer:
                    if in_think:
                        # Look for </think>
                        end_idx = buffer.find("</think>")
                        if end_idx != -1:
                            # Found closing tag — skip everything up to and including it
                            buffer = buffer[end_idx + 8:]
                            in_think = False
                        elif len(buffer) > 8:
                            # Keep last 8 chars as potential partial tag
                            buffer = buffer[-8:]
                            break
                        else:
                            break
                    else:
                        # Look for <think>
                        start_idx = buffer.find("<think>")
                        if start_idx != -1:
                            # Output everything before <think>
                            output += buffer[:start_idx]
                            buffer = buffer[start_idx + 7:]
                            in_think = True
                        elif "<" in buffer:
                            # Potential partial tag — hold back from '<' onward
                            lt_idx = buffer.rfind("<")
                            potential = buffer[lt_idx:]
                            if len(potential) < 7 and "<think>".startswith(potential):
                                output += buffer[:lt_idx]
                                buffer = potential
                                break
                            else:
                                output += buffer
                                buffer = ""
                        else:
                            output += buffer
                            buffer = ""

                if output:
                    yield _make_sse_chunk(output)

    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Cannot connect to llama.cpp server'})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


def _make_sse_chunk(content: str) -> str:
    """Create an SSE chunk with content delta."""
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# =============================================================================
# Non-Streaming Response
# =============================================================================

async def _non_stream_response(llama_body: dict, citations: list, language: str = "de", hyde_text: str = "") -> JSONResponse:
    """Forward request to llama.cpp and post-process the response."""
    try:
        resp = await http_client.post(
            "/v1/chat/completions",
            json=llama_body,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )

        if resp.status_code != 200:
            logger.error(f"llama.cpp returned {resp.status_code}: {resp.text[:500]}")
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"llama.cpp error: {resp.text[:500]}",
            )

        data = resp.json()

        # Post-process: strip <think> tokens and append citations
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

            # Strip <think>...</think> blocks
            if config.enable_think_stripping:
                content = re.sub(
                    r"<think>.*?</think>",
                    "",
                    content,
                    flags=re.DOTALL,
                )
                content = content.strip()

            # Append HyDE expansion section
            if config.enable_hyde_visibility and hyde_text:
                content += _format_hyde_section(hyde_text, language)

            # Append citation footer
            if config.enable_citations and citations:
                content += _format_citation_footer(citations, language)

            choices[0]["message"]["content"] = content

        return JSONResponse(content=data)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="Cannot connect to llama.cpp server",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Non-stream proxy error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=502, detail=f"Proxy error: {e}")


# =============================================================================
# Citation Formatting
# =============================================================================

def _format_hyde_section(hyde_text: str, language: str = "de") -> str:
    """Format the HyDE hypothetical document as a blockquote section."""
    if not hyde_text:
        return ""
    label = "HyDE-Expansion (hypothetisches Dokument)" if language == "de" else "HyDE Expansion (hypothetical document)"
    # Indent each line as blockquote
    quoted = "\n".join(f"> {line}" for line in hyde_text.splitlines())
    return f"\n\n---\n**{label}:**\n{quoted}"


def _format_citation_footer(citations: list, language: str = "de") -> str:
    """Format citation footer for appending to responses.

    Language-aware: uses "Quellen" for German, "Sources" for English.
    Makes citations clickable with intranet URLs when available.
    """
    if language == "de":
        header = "\n\n---\n**Quellen:**"
        source_label = "Quelle"
    else:
        header = "\n\n---\n**Sources:**"
        source_label = "Source"

    lines = [header]
    for c in citations:
        source_name = Path(c.source_file).stem if c.source_file else f"Chunk {c.chunk_id}"
        section_info = f" — {c.section}" if c.section else ""

        if c.intranet_url:
            # Clickable markdown link
            lines.append(
                f"- [{source_label} {c.index}] "
                f"[{source_name}]({c.intranet_url}){section_info}"
            )
        else:
            lines.append(
                f"- [{source_label} {c.index}] {source_name}{section_info}"
            )
    return "\n".join(lines)


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.post("/admin/reindex")
async def reindex(request: Request):
    """Reindex a specific data source or all sources."""
    body = await request.json() if await request.body() else {}
    source = body.get("source", "all")

    if source == "all":
        sources = list(config.data_sources.keys())
    elif source in config.data_sources:
        sources = [source]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source: {source}. Available: {list(config.data_sources.keys())}",
        )

    # Reload indices
    for name in sources:
        ds = config.data_sources[name]
        pipeline._source_manager.load_source(
            name, ds.faiss_index_path, ds.bm25_index_path
        )

    return JSONResponse(content={
        "status": "ok",
        "reindexed": sources,
        "loaded_sources": pipeline.loaded_sources,
    })


@app.get("/admin/stats")
async def stats():
    """Return pipeline statistics."""
    return JSONResponse(content=pipeline.source_stats())


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the RAG server."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced RAG Server")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--log-level", default=None, help="Log level")
    args = parser.parse_args()

    # CLI args override env vars
    if args.host:
        import os
        os.environ["RAG_HOST"] = args.host
    if args.port:
        import os
        os.environ["RAG_PORT"] = str(args.port)
    if args.log_level:
        import os
        os.environ["RAG_LOG_LEVEL"] = args.log_level

    # Read config for host/port
    _config = RAGServerConfig.from_env()

    uvicorn.run(
        "src.inference.rag_server:app",
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
