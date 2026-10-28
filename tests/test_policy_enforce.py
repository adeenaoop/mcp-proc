"""Step 4 tests: enforce mode -- denied calls are really blocked. No LLM, no key.

Run from ~/fyp:  pytest tests/test_policy_enforce.py -v
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "mcpfw"))
from proxy import Proxy, block_response  # noqa: E402
from test_policy_audit import (ATTACK, ATTACKER, POLICY, Call, Events, RecordingLog,  # noqa: E402
                               StubPolicy, StubStdin, StubTee, call_line, run_e2e)
from wire import Tee, WireMismatch, verify_tee  # noqa: E402

ENFORCE = ["--policy", str(POLICY), "--rules", "R1", "--mode", "enforce"]   # R1 tests; R2 tested separately


def attacker_txns(env):
    return [t for t in env["bank_account"]["transactions"] if t["recipient"] == ATTACKER]


# ---------- end to end: agent -> bridge -> mcpfw (enforce) -> server ----------
def test_attack_is_blocked(tmp_path):
    outputs, recs, env = run_e2e(ATTACK, tmp_path, ENFORCE)
    text, is_error = outputs[1]
    # the agent got a tool error naming the rule
    assert is_error is True and text.startswith("Blocked by mcpfw (R1)")
    # the money did NOT move: the server never ran send_money
    assert attacker_txns(env) == []
    # the log
    r0, r1 = recs
    assert (r0["decision"], r0["action"], r0["committed"]) == ("allow", "forward", True)
    assert (r1["decision"], r1["action"], r1["committed"]) == ("deny", "block", False)
    assert r1["is_error"] is True and r1["result_text"] == text and r1["rules_fired"] == ["R1"]
    assert r1["states_before"] == r1["states_after"] == {"R1": "TAINTED"}
    assert r1["mode"] == "enforce" and len(recs) == len(ATTACK)          # X3: one record per call


def test_blocked_call_never_reached_server(tmp_path):
    run_e2e(ATTACK, tmp_path, ENFORCE)
    w = tmp_path / "wire"
    assert b"send_money" not in (w / "up_out.bin").read_bytes()
    assert b"send_money" in (w / "up_blocked.bin").read_bytes()
    assert b"Blocked by mcpfw (R1)" in (w / "down_injected.bin").read_bytes()
    assert b"Blocked by mcpfw" not in (w / "down_in.bin").read_bytes()  # not from the server


def test_wire_accounting_holds_under_enforce(tmp_path):
    run_e2e(ATTACK + [Call("get_balance", {})], tmp_path, ENFORCE)
    verify_tee(tmp_path / "wire")                    # raises WireMismatch if any byte is unaccounted


def test_egress_without_read_still_allowed(tmp_path):
    outputs, recs, env = run_e2e([ATTACK[1]], tmp_path, ENFORCE)
    assert outputs[0][1] is False and len(attacker_txns(env)) == 1
    assert (recs[0]["decision"], recs[0]["action"]) == ("allow", "forward")


def test_session_continues_after_block(tmp_path):
    calls = ATTACK + [Call("get_balance", {}), Call("get_iban", {})]
    outputs, recs, _ = run_e2e(calls, tmp_path, ENFORCE)
    assert [o[1] for o in outputs] == [False, True, False, False]
    assert [r["action"] for r in recs] == ["forward", "block", "forward", "forward"]


def test_audit_mode_unchanged_by_enforce_code(tmp_path):
    outputs, recs, env = run_e2e(ATTACK, tmp_path, ["--policy", str(POLICY), "--rules", "R1"])   # default = audit
    assert outputs[1][1] is False and len(attacker_txns(env)) == 1
    assert (recs[1]["decision"], recs[1]["action"]) == ("would_deny", "forward")


# ---------- block response format ----------
@pytest.mark.parametrize("msg_id", [1, 0, 42, "abc", "1", "x-é"])
def test_block_response_echoes_id_exactly(msg_id):
    result, line = block_response(msg_id, "Blocked by mcpfw (R1): reason")
    assert line.endswith(b"\n") and line.count(b"\n") == 1               # one line
    msg = json.loads(line)
    assert msg["jsonrpc"] == "2.0" and msg["id"] == msg_id and type(msg["id"]) is type(msg_id)
    assert msg["result"] == result
    assert result == {"content": [{"type": "text", "text": "Blocked by mcpfw (R1): reason"}], "isError": True}


def test_block_response_with_newline_in_reason_stays_one_line():
    _, line = block_response(1, "line one\nline two")
    assert line.count(b"\n") == 1 and json.loads(line)["result"]["content"][0]["text"] == "line one\nline two"


# ---------- in-process: fail closed, ordering ----------
class Out:
    def __init__(self, events=None):
        self.data, self.events = b"", events

    def write(self, b):
        self.data += b
        if self.events is not None:
            self.events.append(("respond", json.loads(b)["id"]))

    def flush(self):
        pass


def run_enforce(policy, lines, log=None, events=None):
    log = log or RecordingLog()
    out = Out(events)

    async def go():
        reader = asyncio.StreamReader()
        for line in lines:
            reader.feed_data(line)
        reader.feed_eof()

        class Stdin(StubStdin):
            def write(self, b):
                self.data += b

        class Child:
            stdin = Stdin([])
        proxy = Proxy(log, StubTee(), policy, "enforce", out=out)
        await proxy.up_pump(reader, Child)
        return proxy, Child.stdin
    proxy, stdin = asyncio.run(go())
    return log, proxy, stdin, out


def test_deny_is_not_forwarded_and_not_committed():
    ev = Events()
    log, proxy, stdin, out = run_enforce(StubPolicy(ev, {"send_money": False}), [call_line("send_money", 7)])
    assert stdin.data == b"" and ("commit", "send_money") not in ev
    assert json.loads(out.data)["id"] == 7 and proxy.pending == {}
    assert log.records[0]["extra"]["decision"] == "deny" and log.records[0]["is_error"] is True


def test_policy_error_is_blocked_fail_closed():
    ev = Events()
    log, _, stdin, out = run_enforce(StubPolicy(ev, {}, raise_on_decide=True), [call_line("read_file")])
    assert stdin.data == b""
    extra = log.records[0]["extra"]
    assert (extra["decision"], extra["action"], extra["committed"]) == ("policy_error", "block", False)
    assert "no policy decision" in json.loads(out.data)["result"]["content"][0]["text"]


@pytest.mark.parametrize("bad", [
    {"jsonrpc": "2.0", "id": "a", "method": "tools/call", "params": []},
    {"jsonrpc": "2.0", "id": "b", "method": "tools/call"},
    {"jsonrpc": "2.0", "id": "c", "method": "tools/call", "params": {"name": 5}},
])
def test_malformed_call_is_blocked(bad):
    ev = Events()
    log, _, stdin, out = run_enforce(StubPolicy(ev, {}), [json.dumps(bad).encode() + b"\n"])
    assert stdin.data == b"" and json.loads(out.data)["id"] == bad["id"]
    assert log.records[0]["extra"]["decision"] == "policy_error"
    assert not any(e[0] == "decide" for e in ev)


def test_record_written_before_block_response():
    ev = Events()

    class OrderLog(RecordingLog):
        def write(self, **kw):
            ev.append(("log", kw["tool"]))
            super().write(**kw)
    run_enforce(StubPolicy(ev, {"send_money": False}), [call_line("send_money", 3)], log=OrderLog(), events=ev)
    assert ev.index(("log", "send_money")) < ev.index(("respond", 3))


def test_allowed_calls_still_forwarded_in_enforce():
    ev = Events()
    _, proxy, stdin, out = run_enforce(StubPolicy(ev, {}), [call_line("read_file")])
    assert stdin.data == call_line("read_file") and out.data == b""
    assert ("commit", "read_file") in ev


def test_non_call_messages_untouched_in_enforce():
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode() + b"\n"
    _, _, stdin, out = run_enforce(StubPolicy(Events(), {}), [line])
    assert stdin.data == line and out.data == b""


# ---------- verify_tee catches unaccounted bytes ----------
def write_tee(d, **files):
    t = Tee(d)
    for name, lines in files.items():
        for line in lines:
            t.write(name, line)
    t.close()


def test_verify_tee_accepts_correct_accounting(tmp_path):
    write_tee(tmp_path, up_in=[b"a\n", b"b\n", b"c\n"], up_out=[b"a\n", b"c\n"], up_blocked=[b"b\n"],
              down_in=[b"x\n"], down_out=[b"x\n", b"B\n"], down_injected=[b"B\n"])
    verify_tee(tmp_path)


@pytest.mark.parametrize("files", [
    dict(up_in=[b"a\n", b"b\n"], up_out=[b"a\n"]),                                   # b vanished
    dict(up_in=[b"a\n"], up_out=[b"a\n", b"z\n"]),                                   # z appeared
    dict(up_in=[b"a\n"], up_out=[b"A\n"]),                                           # changed byte
    dict(up_in=[b"a\n", b"b\n"], up_out=[b"b\n", b"a\n"]),                           # reordered
    dict(down_in=[b"x\n"], down_out=[b"x\n", b"evil\n"]),                            # unexplained output
    dict(down_in=[b"x\n"], down_out=[b"x\n"], down_injected=[b"B\n"]),               # injected but never sent
])
def test_verify_tee_rejects_bad_accounting(tmp_path, files):
    write_tee(tmp_path, **files)
    with pytest.raises(WireMismatch):
        verify_tee(tmp_path)
