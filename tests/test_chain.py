"""X3: log integrity (guide §8.7). No LLM, no key.

For every banking user task and injection task, replays its ground_truth()
calls through mcpfw and checks trace.jsonl:
  1. recomputing the hash chain reproduces every h
  2. one record per tool call
     (before the key: = number of replayed calls; with a real episode, the
      guide compares against the number of tool-result messages)
  3. the records match the calls actually made: same tool, same args, same
     result text, same error status, seq 0..n-1 in order
Then tampering checks: changing any single byte, deleting a middle record,
or swapping two records must make verification fail.

Run from ~/fyp:  pytest tests/test_chain.py -v
"""
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent / "mcpfw"))
from events import ChainError, verify_chain  # noqa: E402
from test_replay import BAD_CALLS, numeric, run_direct, suite  # noqa: E402
from test_wire import run_through_proxy  # noqa: E402


def check_trace(trace, calls, outputs):
    records = verify_chain(trace)                                   # 1
    assert len(records) == len(calls), f"{len(records)} records for {len(calls)} calls"   # 2
    for i, (rec, call, (text, is_error)) in enumerate(zip(records, calls, outputs)):       # 3
        assert rec["seq"] == i
        assert rec["tool"] == call.function
        assert rec["args"] == dict(call.args)
        assert rec["result_text"] == text
        assert rec["is_error"] == is_error
        assert rec["t_resp"] >= rec["t_req"]
    return records


@pytest.mark.parametrize("task_id", numeric(suite.user_tasks))
def test_chain_user_task(task_id, tmp_path):
    calls = run_direct(suite.user_tasks[task_id], True)[0]
    outputs = run_through_proxy(calls, tmp_path, task_id, tee=False)
    check_trace(tmp_path / "trace.jsonl", calls, outputs)


@pytest.mark.parametrize("task_id", numeric(suite.injection_tasks))
def test_chain_injection_task(task_id, tmp_path):
    calls = run_direct(suite.injection_tasks[task_id], False)[0]
    outputs = run_through_proxy(calls, tmp_path, tee=False)
    check_trace(tmp_path / "trace.jsonl", calls, outputs)


def test_chain_error_calls(tmp_path):
    outputs = run_through_proxy(BAD_CALLS, tmp_path, tee=False)
    records = check_trace(tmp_path / "trace.jsonl", BAD_CALLS, outputs)
    assert all(r["is_error"] for r in records)


# ---------- tampering: the check must be able to fail ----------
@pytest.fixture(scope="module")
def good_trace(tmp_path_factory):
    """A real trace with several records (the task with the most calls)."""
    tid = max(suite.user_tasks, key=lambda t: len(run_direct(suite.user_tasks[t], True)[0]))
    d = tmp_path_factory.mktemp("tamper")
    calls = run_direct(suite.user_tasks[tid], True)[0]
    run_through_proxy(calls, d, tid, tee=False)
    data = (d / "trace.jsonl").read_bytes()
    assert len(verify_chain(d / "trace.jsonl")) >= 2, "need at least 2 records to test tampering"
    return data


def test_every_single_byte_change_is_detected(good_trace, tmp_path):
    """Change each byte in turn (except newlines) and verify each time."""
    path = tmp_path / "t.jsonl"
    missed = []
    for i, b in enumerate(good_trace):
        if b == ord("\n"):
            continue
        bad = bytearray(good_trace)
        bad[i] = ord("x") if b != ord("x") else ord("y")
        path.write_bytes(bytes(bad))
        try:
            verify_chain(path)
            missed.append(i)
        except ChainError:
            pass
    assert not missed, f"{len(missed)} single-byte changes were NOT detected, e.g. at {missed[:5]}"


def test_deleted_middle_record_is_detected(good_trace, tmp_path):
    lines = good_trace.splitlines(keepends=True)
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"".join(lines[:1] + lines[2:]) if len(lines) > 2 else lines[1])
    with pytest.raises(ChainError):
        verify_chain(path)


def test_swapped_records_are_detected(good_trace, tmp_path):
    lines = good_trace.splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"".join(lines))
    with pytest.raises(ChainError):
        verify_chain(path)
