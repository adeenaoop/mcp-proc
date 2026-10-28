"""Step 2 tests: policy files and the multi-rule Policy. No proxy, no server, no LLM.

Run from ~/fyp:  pytest tests/test_policy_file.py -v
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcpfw"))
from policy import (Policy, PolicyError, StaleDecisionError, TaintThenDeny,  # noqa: E402
                    load_policy, policy_from_dict)

BANKING = ROOT / "policies" / "banking.json"

GOOD = {
    "version": 1,
    "suite": "banking",
    "classes": {
        "untrusted_source": ["read_file", "get_most_recent_transactions"],
        "egress": ["send_money", "schedule_transaction", "update_scheduled_transaction"],
        "privileged": ["update_password", "update_user_info"],
    },
    "rules": {
        "R1": {"type": "taint_then_deny", "source": "untrusted_source", "sink": "egress"},
        "RX": {"type": "taint_then_deny", "source": "untrusted_source", "sink": "privileged"},
    },
}


def good():
    return copy.deepcopy(GOOD)


def step(p, tool, server=None):
    d = p.decide(tool, server)
    if d.allow:
        p.commit(d)
    return d


# ---------- the real banking file ----------
def test_banking_file_loads():
    p = load_policy(BANKING)
    assert isinstance(p, Policy) and p.suite == "banking" and p.enabled == ("R1", "R2")


def test_banking_file_tools_exist_in_agentdojo():
    """Every tool named in the policy must be a real banking tool (catches typos)."""
    from agentdojo.task_suite.load_suites import get_suite
    real = {f.name for f in get_suite("v1.2.2", "banking").tools}
    data = json.loads(BANKING.read_text())
    named = {t for tools in data["classes"].values() for t in tools}
    assert named <= real, f"not real banking tools: {sorted(named - real)}"


def test_file_policy_behaves_like_hand_built_r1():
    data = json.loads(BANKING.read_text())
    hand = TaintThenDeny("R1", data["classes"]["untrusted_source"], data["classes"]["egress"])
    sequences = [
        ["get_balance", "send_money"],
        ["read_file", "send_money"],
        ["send_money", "get_most_recent_transactions", "schedule_transaction", "get_iban"],
        ["get_user_info", "update_password", "read_file", "update_scheduled_transaction"],
    ]
    for seq in sequences:
        p = load_policy(BANKING, ["R1"])                      # R1 only: compared with a hand-built R1
        h = TaintThenDeny("R1", data["classes"]["untrusted_source"], data["classes"]["egress"])
        for tool in seq:
            pd, hd = p.decide(tool), h.decide(tool)
            assert (pd.allow, pd.reason) == (hd.allow, hd.reason), (seq, tool)
            if pd.allow:
                p.commit(pd)
                h.commit(hd)
    assert hand.state == "CLEAN"      # untouched control


# ---------- enabling rules ----------
def test_all_rules_enabled_by_default_in_file_order():
    assert policy_from_dict(good()).enabled == ("R1", "RX")


def test_enable_subset():
    p = policy_from_dict(good(), enabled=["RX"])
    assert p.enabled == ("RX",) and list(p.rules) == ["RX"]


def test_enabled_order_follows_file_not_argument():
    assert policy_from_dict(good(), enabled=["RX", "R1"]).enabled == ("R1", "RX")


@pytest.mark.parametrize("enabled", [[], ["R9"], "R1"])
def test_bad_enabled_rejected(enabled):
    with pytest.raises(PolicyError):
        policy_from_dict(good(), enabled=enabled)


# ---------- several rules together ----------
def test_denied_if_any_rule_denies():
    p = policy_from_dict(good())
    step(p, "read_file")
    d = p.decide("update_password")            # RX denies, R1 allows (not an egress tool)
    assert not d.allow and d.rules_fired == ("RX",)
    assert "(RX)" in d.reason and "(R1)" not in d.reason


def test_all_denying_rules_listed():
    data = good()
    data["classes"]["egress"].append("update_password")          # now both rules deny it
    p = policy_from_dict(data)
    step(p, "read_file")
    d = p.decide("update_password")
    assert d.rules_fired == ("R1", "RX")
    assert d.reason.index("(R1)") < d.reason.index("(RX)")       # file order


def test_allowed_call_commits_every_rule():
    p = policy_from_dict(good())
    step(p, "read_file")
    assert {rid: s["state"] for rid, s in p.snapshot().items()} == {"R1": "TAINTED", "RX": "TAINTED"}


def test_denied_call_changes_no_rule():
    p = policy_from_dict(good())
    step(p, "read_file")
    before = p.snapshot()
    d = p.decide("send_money")
    with pytest.raises(ValueError):
        p.commit(d)
    assert p.snapshot() == before


def test_states_before_and_after():
    p = policy_from_dict(good())
    d = p.decide("read_file")
    assert d.states_before() == {"R1": "CLEAN", "RX": "CLEAN"}
    assert d.states_after() == {"R1": "TAINTED", "RX": "TAINTED"}
    p.commit(d)
    denied = p.decide("send_money")
    assert denied.states_after() == denied.states_before()


def test_stale_policy_decision_rejected_and_nothing_changes():
    p = policy_from_dict(good())
    old = p.decide("send_money")                # allowed while CLEAN
    step(p, "read_file")
    before = p.snapshot()
    with pytest.raises(StaleDecisionError):
        p.commit(old)
    assert p.snapshot() == before


def test_commit_is_all_or_nothing():
    """If only ONE rule's decision is stale, NO rule may change."""
    p = policy_from_dict(good())
    d = p.decide("read_file")
    rx = p.rules["RX"]
    rx.commit(rx.decide("get_balance"))         # RX moves on behind the policy's back
    before = p.snapshot()
    with pytest.raises(StaleDecisionError):
        p.commit(d)
    assert p.snapshot() == before               # R1 was NOT committed either


def test_decision_from_another_policy_rejected():
    a, b = policy_from_dict(good()), policy_from_dict(good())
    with pytest.raises(ValueError):
        b.commit(a.decide("read_file"))


# ---------- fingerprint ----------
def test_fingerprint_is_stable_and_order_independent():
    a = policy_from_dict(good()).fingerprint
    reordered = dict(reversed(list(good().items())))
    assert a == policy_from_dict(good()).fingerprint == policy_from_dict(reordered).fingerprint


def test_fingerprint_changes_with_any_edit():
    data = good()
    data["classes"]["egress"].append("get_iban")
    assert policy_from_dict(data).fingerprint != policy_from_dict(good()).fingerprint


def test_fingerprint_matches_file():
    assert load_policy(BANKING).fingerprint == policy_from_dict(json.loads(BANKING.read_text())).fingerprint


# ---------- bad files are rejected ----------
def mutate(fn):
    data = good()
    fn(data)
    return data


BAD = {
    "not an object": [],
    "missing version": mutate(lambda d: d.pop("version")),
    "wrong version": mutate(lambda d: d.update(version=2)),
    "bool version": mutate(lambda d: d.update(version=True)),
    "missing suite": mutate(lambda d: d.pop("suite")),
    "unknown top key": mutate(lambda d: d.update(rule={})),
    "notes not string": mutate(lambda d: d.update(notes=5)),
    "classes empty": mutate(lambda d: d.update(classes={})),
    "class empty list": mutate(lambda d: d["classes"].update(egress=[])),
    "class is a string": mutate(lambda d: d["classes"].update(egress="send_money")),
    "tool not a string": mutate(lambda d: d["classes"]["egress"].append(5)),
    "duplicate tool": mutate(lambda d: d["classes"]["egress"].append("send_money")),
    "rules empty": mutate(lambda d: d.update(rules={})),
    "rule without type": mutate(lambda d: d["rules"]["R1"].pop("type")),
    "unknown rule type": mutate(lambda d: d["rules"]["R1"].update(type="magic")),
    "rule key typo": mutate(lambda d: d["rules"]["R1"].update(sinks="egress")),
    "rule missing sink": mutate(lambda d: d["rules"]["R1"].pop("sink")),
    "rule uses undefined class": mutate(lambda d: d["rules"]["R1"].update(sink="exfil")),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_policy_rejected(name):
    with pytest.raises(PolicyError):
        policy_from_dict(BAD[name])


def test_duplicate_json_keys_rejected(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"version": 1, "suite": "banking", "classes": {"a": ["x"], "b": ["y"]},'
                 ' "rules": {"R1": {"type": "taint_then_deny", "source": "a", "sink": "b"},'
                 '           "R1": {"type": "taint_then_deny", "source": "b", "sink": "a"}}}')
    with pytest.raises(PolicyError, match="duplicate key"):
        load_policy(f)


def test_invalid_json_rejected(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"version": 1,')
    with pytest.raises(PolicyError, match="invalid JSON"):
        load_policy(f)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(PolicyError, match="cannot read"):
        load_policy(tmp_path / "nope.json")
