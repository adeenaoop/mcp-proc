"""Unit tests for R1 (mcpfw/policy.py). Pure Python: no proxy, no server, no LLM.

Run from ~/fyp:  pytest tests/test_policy_unit.py -v
"""
import sys
from itertools import product
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcpfw"))
from policy import (CLEAN, OTHER, SINK, SOURCE, SOURCE_SINK, TAINTED,  # noqa: E402
                    TRANSITIONS, StaleDecisionError, TaintThenDeny)

# Banking R1. Sources = measured by the canary probe; sinks = policy choice.
BANK_SOURCES = {"read_file", "get_most_recent_transactions"}
BANK_SINKS = {"send_money", "schedule_transaction", "update_scheduled_transaction"}
# Slack R1: get_webpage is both a source and a sink
SLACK_SOURCES = {"get_webpage", "get_channels"}
SLACK_SINKS = {"send_direct_message", "send_channel_message", "post_webpage", "get_webpage"}


def r1():
    return TaintThenDeny("R1", BANK_SOURCES, BANK_SINKS)


def step(rule, tool, server=None):
    """One call the way the proxy will do it: decide, forward if allowed, commit."""
    d = rule.decide(tool, server)
    if d.allow:
        rule.commit(d)
    return d


def run(rule, tools):
    return [step(rule, t).allow for t in tools]


# ---------- the DFA table ----------
def test_table_is_complete():
    for state, cls in product([CLEAN, TAINTED], [SOURCE, SINK, SOURCE_SINK, OTHER]):
        assert (state, cls) in TRANSITIONS


TOOL_FOR = {SOURCE: "read_file", SINK: "send_money", SOURCE_SINK: "both_tool", OTHER: "get_balance"}
CELLS = [
    (CLEAN, SOURCE, True, TAINTED),
    (CLEAN, SINK, True, CLEAN),
    (CLEAN, SOURCE_SINK, True, TAINTED),
    (CLEAN, OTHER, True, CLEAN),
    (TAINTED, SOURCE, True, TAINTED),
    (TAINTED, SINK, False, TAINTED),
    (TAINTED, SOURCE_SINK, False, TAINTED),
    (TAINTED, OTHER, True, TAINTED),
]


def rule_in(state):
    rule = TaintThenDeny("R1", BANK_SOURCES | {"both_tool"}, BANK_SINKS | {"both_tool"})
    if state == TAINTED:
        step(rule, "read_file")
    assert rule.state == state
    return rule


@pytest.mark.parametrize("state,cls,allow,after", CELLS)
def test_every_cell_decision(state, cls, allow, after):
    d = rule_in(state).decide(TOOL_FOR[cls])
    assert (d.tool_class, d.state_before, d.allow, d.state_after) == (cls, state, allow, after)


@pytest.mark.parametrize("state,cls,allow,after", CELLS)
def test_every_cell_commit(state, cls, allow, after):
    """commit() really applies the transition (allowed cells) or is refused (denied cells)."""
    rule = rule_in(state)
    version_before, snap_before = rule.version, rule.snapshot()
    d = rule.decide(TOOL_FOR[cls])
    if allow:
        rule.commit(d)
        assert rule.state == after
        assert rule.version == version_before + 1
        tainted_by_this = cls in (SOURCE, SOURCE_SINK)
        assert len(rule.taints) == len(snap_before["taints"]) + tainted_by_this
    else:
        with pytest.raises(ValueError):
            rule.commit(d)
        assert rule.snapshot() == snap_before          # denied cell leaves state untouched


# ---------- behaviour on call sequences ----------
def test_egress_without_prior_read_is_allowed():
    assert run(r1(), ["get_balance", "send_money"]) == [True, True]


def test_read_then_egress_is_denied():
    # banking injection_task_0's goal: money out after reading attacker text
    assert run(r1(), ["read_file", "send_money"]) == [True, False]


def test_every_sink_denied_after_taint():
    rule = r1()
    step(rule, "get_most_recent_transactions")
    for sink in BANK_SINKS:
        assert rule.decide(sink).allow is False


def test_stays_denied_every_time():
    assert run(r1(), ["read_file", "send_money", "send_money", "schedule_transaction"]) == [True, False, False, False]


def test_other_tools_do_not_taint():
    assert run(r1(), ["get_balance", "get_iban", "get_user_info", "send_money"]) == [True, True, True, True]


def test_unknown_tool_is_allowed_and_does_not_taint():
    assert run(r1(), ["not_a_real_tool", "send_money"]) == [True, True]


def test_reads_are_never_denied():
    assert run(r1(), ["read_file", "read_file", "get_most_recent_transactions"]) == [True, True, True]


def test_egress_before_read_then_denied_after():
    assert run(r1(), ["send_money", "read_file", "send_money"]) == [True, True, False]


# ---------- decide / commit lifecycle ----------
def test_decide_never_changes_state():
    rule = r1()
    for _ in range(3):
        rule.decide("read_file")
    assert rule.state == CLEAN and rule.taints == [] and rule.version == 0


def test_denied_call_does_not_change_state():
    rule = r1()
    step(rule, "read_file")
    before = rule.snapshot()
    rule.decide("send_money")
    assert rule.snapshot() == before


def test_cannot_commit_a_denied_decision():
    rule = r1()
    step(rule, "read_file")
    with pytest.raises(ValueError):
        rule.commit(rule.decide("send_money"))
    assert rule.state == TAINTED


def test_stale_decision_rejected():
    """Decide on send_money while CLEAN, then the state changes: the old 'allow' must not be usable."""
    rule = r1()
    old = rule.decide("send_money")                 # allowed, made while CLEAN
    step(rule, "read_file")                         # state changes to TAINTED
    with pytest.raises(StaleDecisionError):
        rule.commit(old)
    assert rule.state == TAINTED and rule.version == 1


def test_stale_decision_rejected_even_if_state_name_unchanged():
    """State stays TAINTED, but a commit happened in between: still stale (version check)."""
    rule = r1()
    step(rule, "read_file")
    old = rule.decide("get_balance")
    step(rule, "get_iban")                          # TAINTED -> TAINTED, version moves on
    with pytest.raises(StaleDecisionError):
        rule.commit(old)


def test_same_decision_cannot_be_committed_twice():
    rule = r1()
    d = rule.decide("read_file")
    rule.commit(d)
    with pytest.raises(StaleDecisionError):
        rule.commit(d)
    assert len(rule.taints) == 1


def test_decision_from_another_rule_rejected():
    a, b = r1(), r1()
    with pytest.raises(ValueError):
        b.commit(a.decide("read_file"))
    assert b.state == CLEAN


def test_commit_requires_a_decision():
    with pytest.raises(TypeError):
        r1().commit("read_file")


def test_commit_records_server_from_decision():
    rule = r1()
    step(rule, "read_file", server="files")
    assert rule.snapshot()["taints"] == [{"tool": "read_file", "server": "files"}]


# ---------- tool that is both source and sink (Slack get_webpage) ----------
def test_source_sink_tool():
    rule = TaintThenDeny("R1", SLACK_SOURCES, SLACK_SINKS)
    # first visit allowed and taints; second denied (by tool class, not URL content)
    assert run(rule, ["get_webpage", "get_webpage"]) == [True, False]


def test_slack_exfiltration_sequence_denied():
    rule = TaintThenDeny("R1", SLACK_SOURCES, SLACK_SINKS)
    assert run(rule, ["read_channel_messages", "get_webpage", "post_webpage"]) == [True, True, False]


# ---------- deny reason ----------
def test_reason_names_rule_tool_and_first_source():
    rule = r1()
    step(rule, "get_most_recent_transactions")
    step(rule, "read_file")
    d = rule.decide("send_money")
    assert "(R1)" in d.reason and "`send_money`" in d.reason
    assert "`get_most_recent_transactions`" in d.reason


def test_reason_does_not_claim_leak_detection():
    rule = r1()
    step(rule, "read_file")
    reason = rule.decide("send_money").reason.lower()
    assert "egress-class" in reason and "leak" not in reason


def test_allowed_calls_have_empty_reason():
    assert r1().decide("send_money").reason == ""


def test_deterministic():
    seq = ["get_balance", "read_file", "send_money", "get_iban", "schedule_transaction"]
    a, b = r1(), r1()
    assert [step(a, t) for t in seq] == [step(b, t) for t in seq]


# ---------- server signal: diagnostic only ----------
def test_cross_server_signal():
    rule = r1()
    step(rule, "read_file", server="files")
    assert rule.decide("send_money", server="bank").cross_server is True
    assert rule.decide("send_money", server="files").cross_server is False
    assert rule.decide("send_money").cross_server is None
    assert rule.decide("send_money", server="bank").allow is False     # never changes the decision


def test_cross_server_unknown_if_any_taint_lacks_server():
    rule = r1()
    step(rule, "read_file")
    assert rule.decide("send_money", server="bank").cross_server is None


def test_cross_server_not_reported_for_non_sinks():
    rule = r1()
    step(rule, "read_file", server="files")
    assert rule.decide("get_balance", server="bank").cross_server is None


def test_reason_mentions_server_when_known():
    rule = r1()
    step(rule, "read_file", server="files")
    assert "on server `files`" in rule.decide("send_money", server="bank").reason


# ---------- configuration errors fail loudly ----------
@pytest.mark.parametrize("kwargs,err", [
    ({"rule_id": "", "sources": {"a"}, "sinks": {"b"}}, ValueError),
    ({"rule_id": "R1", "sources": set(), "sinks": {"b"}}, ValueError),
    ({"rule_id": "R1", "sources": {"a"}, "sinks": set()}, ValueError),
    ({"rule_id": "R1", "sources": "read_file", "sinks": {"b"}}, TypeError),   # string, not a set
    ({"rule_id": "R1", "sources": {"a", 5}, "sinks": {"b"}}, TypeError),
])
def test_bad_configuration_rejected(kwargs, err):
    with pytest.raises(err):
        TaintThenDeny(**kwargs)
