"""Fail-closed fixes after step 4 review. No LLM, no key.

  1 arguments must be an object (null = no arguments)
  2 a failing policy.states() never crashes the proxy or lets a call through
  3 a failed commit marks the policy FAILED: enforce blocks every later call
  4 a failed audit-log write stops the session (exit 3); nothing unlogged happens
  5 request ids must be string or integer; invalid ids get JSON-RPC Invalid Request

Run from ~/fyp:  pytest tests/test_policy_failclosed.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "mcpfw"))
from proxy import split_call, valid_request_id  # noqa: E402
from test_policy_audit import POLICY, PROXY, Events, StubPolicy, call_line  # noqa: E402
from test_policy_enforce import run_enforce  # noqa: E402


def close_stdin(p):
    """Close the child's stdin ourselves, then hide it from communicate().

    Python 3.12.x communicate() flushes stdin first and raises ValueError if
    it is already closed (3.13 ignores that). Setting p.stdin = None is how
    subprocess itself marks "no stdin to manage", so this works on both.
    """
    if p.stdin is None:
        return
    try:
        p.stdin.close()
    except BrokenPipeError:            # leftover bytes, but mcpfw stopped reading: fine
        pass
    p.stdin = None


def raw_call(msg_id=1, name="send_money", **params_extra):
    params = {"name": name, **params_extra}
    return json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call", "params": params}).encode() + b"\n"


def run_audit(policy, lines, log=None):
    """Same helper as run_enforce, but in audit mode."""
    import test_policy_enforce as te
    from proxy import Proxy
    orig = Proxy.__init__

    def audit_init(self, log, tee, policy=None, mode="audit", out=None):
        orig(self, log, tee, policy, "audit", out)
    Proxy.__init__ = audit_init
    try:
        return te.run_enforce(policy, lines, log=log)
    finally:
        Proxy.__init__ = orig


# ---------- 1. arguments ----------
@pytest.mark.parametrize("bad_args", [[], "text", 42, True])
def test_non_object_arguments_is_a_problem(bad_args):
    msg = json.loads(raw_call(arguments=bad_args))
    _, _, problem = split_call(msg)
    assert problem and "malformed arguments" in problem


@pytest.mark.parametrize("ok_args", [None, {}, {"a": 1}])
def test_null_or_object_arguments_accepted(ok_args):
    msg = json.loads(raw_call(arguments=ok_args))
    tool, args, problem = split_call(msg)
    assert problem is None and isinstance(args, dict)


@pytest.mark.parametrize("bad_args", [[], "text", 42])
def test_bad_arguments_blocked_in_enforce(bad_args):
    ev = Events()
    log, _, stdin, out = run_enforce(StubPolicy(ev, {}), [raw_call(arguments=bad_args)])
    assert stdin.data == b"" and log.records[0]["extra"]["decision"] == "policy_error"
    assert not any(e[0] == "decide" for e in ev)


@pytest.mark.parametrize("bad_args", [[], "text", 42])
def test_bad_arguments_forwarded_in_audit(bad_args):
    line = raw_call(arguments=bad_args)
    _, proxy, stdin, out = run_audit(StubPolicy(Events(), {}), [line])
    assert stdin.data == line and out.data == b""
    assert next(iter(proxy.pending.values()))[0].extra["decision"] == "policy_error"


# ---------- 2. policy.states() failing ----------
class BrokenStatesPolicy(StubPolicy):
    def states(self):
        raise RuntimeError("state unreadable")


def test_states_failure_blocks_and_never_crashes():
    ev = Events()
    log, proxy, stdin, out = run_enforce(BrokenStatesPolicy(ev, {}), [call_line("read_file", 1),
                                                                     call_line("get_balance", 2)])
    assert stdin.data == b"", "a call went through although policy state could not be read"
    assert [json.loads(l)["id"] for l in out.data.splitlines()] == [1, 2]   # both answered
    extra = log.records[0]["extra"]
    assert extra["decision"] == "policy_error" and "state unreadable" in extra["states_error"]
    assert extra["policy_health"] == "failed" and proxy.policy_failure is not None


# ---------- 3. failed commit -> policy FAILED ----------
def test_failed_commit_blocks_every_later_call_in_enforce():
    ev = Events()
    log, proxy, stdin, out = run_enforce(StubPolicy(ev, {}, raise_on_commit=True),
                                         [call_line("read_file", 1), call_line("get_balance", 2)])
    # the first call WAS sent (its commit failed afterwards): logged as forwarded, not blocked
    assert stdin.data == call_line("read_file", 1)
    first = proxy.pending['1'][0].extra
    assert (first["action"], first["committed"], first["policy_health"]) == ("forward", False, "failed")
    assert "commit boom" in first["commit_error"]
    # the second call is blocked because the policy state can no longer be trusted
    second = log.records[0]["extra"]
    assert (second["decision"], second["action"]) == ("policy_error", "block")
    assert "marked FAILED" in second["policy_error"]
    assert json.loads(out.data)["id"] == 2
    assert [e for e in ev if e[0] == "decide"] == [("decide", "read_file")]   # never asked again


def test_failed_commit_in_audit_logs_policy_error_and_forwards():
    _, proxy, stdin, _ = run_audit(StubPolicy(Events(), {}, raise_on_commit=True),
                                   [call_line("read_file", 1), call_line("get_balance", 2)])
    assert stdin.data == call_line("read_file", 1) + call_line("get_balance", 2)
    assert proxy.pending['2'][0].extra["decision"] == "policy_error"


# ---------- 4. audit log failure -> session stops ----------
class FailingLog:
    def __init__(self):
        self.calls = 0

    def write(self, **kw):
        self.calls += 1
        raise OSError("disk full")


def test_log_failure_on_block_sends_nothing_and_stops():
    ev = Events()
    _, proxy, stdin, out = run_enforce(StubPolicy(ev, {"send_money": False}),
                                       [call_line("send_money", 1), call_line("read_file", 2)],
                                       log=FailingLog())
    assert stdin.data == b"", "denied request reached the server"
    assert out.data == b"", "agent was answered although the block was not logged"
    assert proxy.fatal and "disk full" in proxy.fatal
    assert ("decide", "read_file") not in ev, "session kept running after the log failed"


FAKE_SERVER = """
import json, sys
seen = open(sys.argv[1], "ab")
for line in sys.stdin:
    seen.write(line.encode()); seen.flush()
    msg = json.loads(line)
    if msg.get("method") == "tools/call":
        res = {"content": [{"type": "text", "text": "ok"}], "isError": False}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}) + "\\n")
        sys.stdout.flush()
"""


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs Linux /dev/full")
def test_real_log_failure_exits_3_and_stops_forwarding(tmp_path):
    """/dev/full refuses every write: a real, deterministic log failure.
    The client behaves like AgentDojo's agent: one call, wait for the reply, next call."""
    server, seen = tmp_path / "fake.py", tmp_path / "seen.jsonl"
    server.write_text(FAKE_SERVER)
    p = subprocess.Popen([sys.executable, str(PROXY), "--log", "/dev/full", "--policy", str(POLICY),
                          "--mode", "enforce", "--", sys.executable, str(server), str(seen)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(call_line("read_file", 1))
    p.stdin.flush()
    assert json.loads(p.stdout.readline())["id"] == 1        # the reply arrives; its record fails
    try:                                                      # the agent sends its next call
        p.stdin.write(call_line("get_iban", 2))
        p.stdin.flush()
    except BrokenPipeError:
        pass                                                  # mcpfw already stopped reading: fine
    close_stdin(p)
    _, err = p.communicate(timeout=60)
    assert p.returncode == 3, err
    assert b"audit log write failed" in err
    assert b"get_iban" not in seen.read_bytes(), "forwarded a call after the log failure was detected"


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs Linux /dev/full")
def test_pipelined_calls_sent_before_failure_are_not_recalled(tmp_path):
    """Documents the limit: calls already forwarded BEFORE the failure could be
    detected (pipelined input) cannot be pulled back. Only later calls are stopped."""
    server, seen = tmp_path / "fake.py", tmp_path / "seen.jsonl"
    server.write_text(FAKE_SERVER)
    stdin = call_line("read_file", 1) + call_line("get_balance", 2)
    p = subprocess.run([sys.executable, str(PROXY), "--log", "/dev/full", "--policy", str(POLICY),
                        "--mode", "enforce", "--", sys.executable, str(server), str(seen)],
                       input=stdin, capture_output=True, timeout=60)
    assert p.returncode == 3                                  # still fails closed afterwards


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs Linux /dev/full")
def test_real_log_failure_on_blocked_call(tmp_path):
    server, seen = tmp_path / "fake.py", tmp_path / "seen.jsonl"
    server.write_text(FAKE_SERVER)
    bad = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []}).encode() + b"\n"
    p = subprocess.run([sys.executable, str(PROXY), "--log", "/dev/full", "--policy", str(POLICY),
                        "--mode", "enforce", "--", sys.executable, str(server), str(seen)],
                       input=bad + call_line("get_iban", 2), capture_output=True, timeout=60)
    assert p.returncode == 3
    assert p.stdout == b"", "agent answered without a logged record"
    assert seen.read_bytes() == b"", "server received something after a blocked call's log failed"


# ---------- 5. request ids ----------
@pytest.mark.parametrize("good", [0, 1, -5, 2**63, "x", "", "1"])
def test_valid_ids(good):
    assert valid_request_id(good)


@pytest.mark.parametrize("bad", [None, True, False, 1.5, 100.0, [1], {"a": 1}])
def test_invalid_ids(bad):
    assert not valid_request_id(bad)


def test_float_spelled_id_is_invalid():
    """1e2 parses as the float 100.0: not an integer, so not a valid MCP id."""
    line = b'{"jsonrpc":"2.0","id":1e2,"method":"tools/call","params":{"name":"read_file"}}\n'
    _, _, problem = split_call(json.loads(line))
    assert problem and "invalid request id" in problem


@pytest.mark.parametrize("bad", [None, True, 1.5, [1], {"a": 1}])
def test_invalid_id_blocked_with_invalid_request_error(bad):
    ev = Events()
    log, _, stdin, out = run_enforce(StubPolicy(ev, {}), [raw_call(msg_id=bad, name="read_file")])
    assert stdin.data == b""
    resp = json.loads(out.data)
    assert resp["id"] is None and resp["error"]["code"] == -32600 and "result" not in resp
    assert log.records[0]["extra"]["decision"] == "policy_error"
    assert not any(e[0] == "decide" for e in ev)


@pytest.mark.parametrize("bad", [None, 1.5, [1]])
def test_invalid_id_forwarded_and_flagged_in_audit(bad):
    line = raw_call(msg_id=bad, name="read_file")
    _, proxy, stdin, _ = run_audit(StubPolicy(Events(), {}), [line])
    assert stdin.data == line
    entry = next(iter(proxy.pending.values()))[0]
    assert entry.extra["decision"] == "policy_error" and "invalid request id" in entry.extra["policy_error"]


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs Linux /dev/full")
def test_log_failure_terminates_even_if_agent_stays_silent(tmp_path):
    """After a log failure mcpfw must stop on its own, not wait for the agent's
    next line: the agent keeps its connection open and sends nothing more."""
    server, seen = tmp_path / "fake.py", tmp_path / "seen.jsonl"
    server.write_text(FAKE_SERVER)
    p = subprocess.Popen([sys.executable, str(PROXY), "--log", "/dev/full", "--policy", str(POLICY),
                          "--mode", "enforce", "--", sys.executable, str(server), str(seen)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p.stdin.write(call_line("read_file", 1))
        p.stdin.flush()
        p.stdout.readline()                       # reply arrives; its record fails
        assert p.wait(timeout=20) == 3            # stdin still open: must exit by itself
    finally:
        if p.poll() is None:
            p.kill()
        close_stdin(p)
        p.communicate()
