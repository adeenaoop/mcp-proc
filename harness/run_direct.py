"""Direct-mode runs (guide §3), AgentDojo 0.1.35.

Runs the first N user tasks of a suite (numeric ID order) through AgentDojo's
standard pipeline, with tools executed as normal Python functions (no MCP).
Saves one JSON per episode to run/direct/<task_id>_r<rep>.json.

Usage (from ~/fyp, venv active):
    python harness/run_direct.py --fake            # plumbing test, no key, no cost
    python harness/run_direct.py                   # real run: 5 banking tasks, 1 rep
    python harness/run_direct.py --reps 3          # X4 direct side: 5 tasks x 3 reps

Model and key come from .env:
    LLM_API_KEY=...        (falls back to OPENAI_API_KEY)
    LLM_BASE_URL=...       (optional; e.g. https://openrouter.ai/api/v1)
    LLM_MODEL=...          (default gpt-4o-mini-2024-07-18;
                            on OpenRouter use openai/gpt-4o-mini)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

from agentdojo.agent_pipeline import (
    AgentPipeline, InitQuery, OpenAILLM, SystemMessage, ToolsExecutionLoop, ToolsExecutor,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.task_suite.load_suites import get_suite

from counting_client import CountingClient, CountingRuntime

ROOT = Path(__file__).resolve().parent.parent   # ~/fyp


# ---------- fake client: tests the plumbing without a key ----------
class _FakeCompletions:
    """Answers every request with a short text reply and made-up usage.
    The agent calls no tools, so utility will be False; that is expected.
    The point is to prove the pipeline, counters and JSON writing work."""

    def create(self, **kwargs):
        return ChatCompletion.model_validate({
            "id": "fake", "object": "chat.completion", "created": 0, "model": kwargs.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "FAKE RESPONSE"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
        })


class _FakeChat:
    completions = _FakeCompletions()


class FakeOpenAI:
    chat = _FakeChat()
# -------------------------------------------------------------------


def make_client(fake):
    if fake:
        return FakeOpenAI(), "fake-model"
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("No key: set LLM_API_KEY (or OPENAI_API_KEY) in ~/fyp/.env, or use --fake.")
    base_url = os.environ.get("LLM_BASE_URL") or None
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini-2024-07-18")
    # Generous retries: long experiment runs hit rate limits; the client backs off and retries.
    return OpenAI(api_key=key, base_url=base_url, max_retries=10, timeout=120), model


def build_pipeline(client, model):
    """§3.2: the documented composition, in this order."""
    llm = OpenAILLM(client, model, temperature=0.0)
    pipeline = AgentPipeline([
        SystemMessage(load_system_message(None)),   # AgentDojo's default system prompt
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ])
    # AgentDojo's attacks address the model by name, read from pipeline.name
    # (e.g. tool_knowledge needs "gpt-4o-mini-2024-07-18" -> "GPT-4").
    pipeline.name = model
    return pipeline


def numeric_task_ids(suite):
    """user_task_0, 1, 2 ... (text sort would give 0, 1, 10, 11 ...)."""
    return sorted(suite.user_tasks, key=lambda k: int(k.rsplit("_", 1)[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--bench", default="v1.2.2")
    ap.add_argument("--tasks", type=int, default=5, help="first N user tasks")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--fake", action="store_true", help="no key, no cost: plumbing test")
    args = ap.parse_args()

    suite = get_suite(args.bench, args.suite)
    real_client, model = make_client(args.fake)
    client = CountingClient(real_client)
    pipeline = build_pipeline(client, model)

    out_dir = ROOT / "run" / ("direct_fake" if args.fake else "direct")
    out_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(args.reps):
        for tid in numeric_task_ids(suite)[: args.tasks]:
            client.reset()
            CountingRuntime.reset()
            t0 = time.monotonic()
            error = None
            try:
                utility, _security = suite.run_task_with_pipeline(
                    pipeline, suite.user_tasks[tid],
                    injection_task=None, injections={},
                    runtime_class=CountingRuntime,
                )
            except Exception as e:          # record the failure, keep going
                utility, error = False, f"{type(e).__name__}: {e}"
            record = {
                "mode": "direct", "suite": args.suite, "bench_version": args.bench,
                "model": model, "task": tid, "rep": rep,
                "utility": bool(utility),
                "llm_calls": client.llm_calls, "tok_in": client.tok_in, "tok_out": client.tok_out,
                "n_tool_calls": CountingRuntime.n_tool_calls,
                "wall_s": round(time.monotonic() - t0, 2),
                "error": error,
            }
            (out_dir / f"{tid}_r{rep}.json").write_text(json.dumps(record, indent=2))
            print(f"{tid} r{rep}: utility={record['utility']} calls={record['llm_calls']} "
                  f"tok_in={record['tok_in']} tok_out={record['tok_out']} "
                  f"tools={record['n_tool_calls']} {record['wall_s']}s"
                  + (f"  ERROR {error}" if error else ""))


if __name__ == "__main__":
    main()
