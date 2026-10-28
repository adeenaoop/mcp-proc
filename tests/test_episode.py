"""Tests for §6 run_episode.py. No LLM, no key.

Uses fake models instead of a real LLM:
  - "gt":   requests each task's ground-truth tool calls, then gives the right answer
  - "lazy": calls no tools and answers "I am done."
For every banking user task and both fakes, checks that the MCP episode
runner (agent -> MCPRuntime -> bridge -> mcpfw -> server, scored on the
server's environments) gives the SAME utility as AgentDojo's own
run_task_with_pipeline in direct mode. Also checks the gt fake scores True,
the trace has one record per tool call, and the episode files exist.

Run from ~/fyp:  pytest tests/test_episode.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from counting_client import CountingClient, CountingRuntime  # noqa: E402
from fake_llm import GroundTruthFakeClient  # noqa: E402
from run_direct import build_pipeline, numeric_task_ids  # noqa: E402
from run_episode import run_episode  # noqa: E402

from agentdojo.task_suite.load_suites import get_suite

BENCH, SUITE = "v1.2.2", "banking"
suite = get_suite(BENCH, SUITE)


def fake(task, kind):
    if kind == "gt":
        env = task.init_environment(suite.load_and_inject_default_environment({}))
        return GroundTruthFakeClient(task.ground_truth(env), task.GROUND_TRUTH_OUTPUT)
    return GroundTruthFakeClient([], "I am done.")


@pytest.mark.parametrize("kind", ["gt", "lazy"])
@pytest.mark.parametrize("task_id", numeric_task_ids(suite))
def test_episode_matches_agentdojo(task_id, kind, tmp_path):
    task = suite.user_tasks[task_id]
    r = run_episode(suite, SUITE, BENCH, task_id, 0, CountingClient(fake(task, kind)), "fake", tmp_path)

    direct_utility, _ = suite.run_task_with_pipeline(
        build_pipeline(CountingClient(fake(task, kind)), "fake"), task, None, {},
        runtime_class=CountingRuntime)

    assert r["utility"] == direct_utility, "MCP runner scored differently from AgentDojo"
    if kind == "gt":
        assert r["utility"] is True
    assert r["trace_matches_tool_calls"], f"{r['trace_records']} trace records vs {r['n_tool_calls']} tool calls"

    ep = tmp_path / f"{SUITE}_{task_id}_none_r0"
    for name in ["injections.json", "env_pre.json", "env_post.json", "trace.jsonl", "messages.json", "result.json"]:
        assert (ep / name).exists(), f"missing {name}"
