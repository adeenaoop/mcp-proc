"""R2 in policy files (step 2): format, validation, and Policy.decide with arguments.

Run from ~/fyp:  pytest tests/test_policy_r2_file.py -v
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcpfw"))
from policy import (DistinctDestinationLimit, PolicyError, StaleDecisionError,  # noqa: E402
                    load_policy, policy_from_dict)

FILES = {"banking": ROOT / "policies" / "banking.json", "slack": ROOT / "policies" / "slack.json"}
K = {"banking": 1, "slack": 2}


def data(suite="banking"):
    return json.loads(FILES[suite].read_text())


def with_r2(**changes):
    """The banking file with R2's spec changed (None deletes a key)."""
    d = data()
    for key, value in changes.items():
        if value is None:
            d["rules"]["R2"].pop(key, None)
        else:
            d["rules"]["R2"][key] = value
    return d


def pay(to):
    return {"recipient": to, "amount": 10, "subject": "x", "date": "2022-01-01"}


# ---------- the real files ----------
@pytest.mark.parametrize("suite", FILES)
def test_real_file_loads_r1_and_r2(suite):
    p = load_policy(FILES[suite])
    assert p.enabled == ("R1", "R2")
    r2 = p.rules["R2"]
    assert isinstance(r2, DistinctDestinationLimit) and r2.k == K[suite]
    assert r2.sinks == frozenset(data(suite)["classes"]["egress"])      # same tools as R1 guards


@pytest.mark.parametrize("suite", FILES)
def test_destination_args_match_agentdojo_schemas(suite):
    """Each destination argument really exists on that tool, and `optional` lists
    exactly the tools where it is not required. A typo here would make R2 deny
    every call to that tool as 'missing destination'."""
    from agentdojo.task_suite.load_suites import get_suite
    schemas = {t.name: t.parameters.model_json_schema() for t in get_suite("v1.2.2", suite).tools}
    spec = data(suite)["rules"]["R2"]
    for tool, arg in spec["destination_arg"].items():
        assert arg in schemas[tool]["properties"], f"{tool} has no argument {arg!r}"
    not_required = {t for t, a in spec["destination_arg"].items() if a not in schemas[t].get("required", [])}
    assert set(spec.get("optional", [])) == not_required


@pytest.mark.parametrize("suite", FILES)
def test_r1_alone_still_selectable(suite):
    p = load_policy(FILES[suite], ["R1"])
    assert p.enabled == ("R1",) and "R2" not in p.rules


def test_r2_alone_selectable():
    p = load_policy(FILES["banking"], ["R2"])
    assert p.enabled == ("R2",)
    assert p.named_tools() == set(data()["classes"]["egress"])


# ---------- validation: every mistake is rejected ----------
@pytest.mark.parametrize("k", [0, -1, True, False, "1", 1.5, None, [1]])
def test_bad_k(k):
    d = with_r2(k=k) if k is not None else with_r2(k=None)
    with pytest.raises(PolicyError, match="k"):
        policy_from_dict(d)


@pytest.mark.parametrize("dest, match", [
    ({"send_money": "recipient", "schedule_transaction": "recipient"}, "missing"),       # a sink tool left out
    ({**data()["rules"]["R2"]["destination_arg"], "get_balance": "x"}, "not in the class"),
    ({}, "non-empty"),
    (["send_money"], "non-empty"),
    ({"send_money": "", "schedule_transaction": "recipient",
      "update_scheduled_transaction": "recipient"}, "invalid entries"),
    ({"send_money": 5, "schedule_transaction": "recipient",
      "update_scheduled_transaction": "recipient"}, "invalid entries"),
])
def test_bad_destination_arg(dest, match):
    with pytest.raises(PolicyError, match=match):
        policy_from_dict(with_r2(destination_arg=dest))


@pytest.mark.parametrize("opt, match", [
    ("update_scheduled_transaction", "list"),
    (["get_balance"], "not in class"),
    (["update_scheduled_transaction", "update_scheduled_transaction"], "duplicate"),
    ([""], "list"),
])
def test_bad_optional(opt, match):
    with pytest.raises(PolicyError, match=match):
        policy_from_dict(with_r2(optional=opt))


def test_optional_key_may_be_absent():
    p = policy_from_dict(with_r2(optional=None))
    assert p.rules["R2"].optional == frozenset()


@pytest.mark.parametrize("missing", ["sink", "k", "destination_arg"])
def test_missing_required_key(missing):
    with pytest.raises(PolicyError, match="missing"):
        policy_from_dict(with_r2(**{missing: None}))


def test_unknown_key_rejected():
    with pytest.raises(PolicyError, match="unknown keys"):
        policy_from_dict(with_r2(limit=2))


def test_r1_key_on_r2_rejected():
    """`source` belongs to R1's type; on R2 it is a mistake, not ignored."""
    with pytest.raises(PolicyError, match="unknown keys"):
        policy_from_dict(with_r2(source="untrusted_source"))


def test_undefined_sink_class():
    with pytest.raises(PolicyError, match="not defined"):
        policy_from_dict(with_r2(sink="exfil"))


def test_r1_still_validated():
    d = data()
    d["rules"]["R1"]["source"] = "nope"
    with pytest.raises(PolicyError, match="not defined"):
        policy_from_dict(d)


# ---------- Policy.decide with arguments ----------
def step(p, tool, args=None):
    d = p.decide(tool, args=args)
    if d.allow:
        p.commit(d)
    return d


def test_slide_example_through_policy():
    d = data("slack")
    p = policy_from_dict(d, ["R2"])
    msgs = [step(p, "send_direct_message", {"recipient": x, "body": "hi"}) for x in "AABC"]
    assert [m.allow for m in msgs] == [True, True, True, False]
    assert msgs[3].rules_fired == ("R2",) and "R2" in msgs[3].reason
    assert msgs[3].states_before() == {"R2": "COUNT_2"}


def test_r1_and_r2_together_banking():
    p = load_policy(FILES["banking"])
    first = step(p, "send_money", pay("Landlord"))
    assert first.allow and first.states_after() == {"R1": "CLEAN", "R2": "COUNT_1"}
    second = step(p, "send_money", pay("ATTACKER"))                # new IBAN, k=1: R2 denies, R1 allows (clean)
    assert not second.allow and second.rules_fired == ("R2",)
    step(p, "read_file", {"file_path": "bill.txt"})                # taints R1
    third = step(p, "send_money", pay("ATTACKER"))
    assert third.rules_fired == ("R1", "R2")                       # both rules give their reason
    assert "(R1)" in third.reason and "(R2)" in third.reason


def test_without_args_r2_fails_closed():
    """Until the proxy passes arguments (step 3), an enabled R2 blocks egress
    rather than silently allowing it."""
    p = load_policy(FILES["banking"], ["R2"])
    d = p.decide("send_money")
    assert not d.allow and "cannot be checked" in d.reason
    assert p.decide("get_balance").allow                           # non-egress unaffected


def test_r1_ignores_args():
    p = load_policy(FILES["banking"], ["R1"])
    assert p.decide("send_money", args=None).allow
    assert p.decide("send_money", args=pay("X")).allow


def test_commit_is_all_or_nothing_with_r2():
    p = load_policy(FILES["banking"])
    d1 = p.decide("send_money", args=pay("A"))
    d2 = p.decide("send_money", args=pay("B"))                     # decided at COUNT_0 too
    p.commit(d1)
    with pytest.raises(StaleDecisionError):
        p.commit(d2)
    assert p.states() == {"R1": "CLEAN", "R2": "COUNT_1"}
    assert p.rules["R2"].destinations == ["a"]


def test_file_not_modified_by_loading():
    before = copy.deepcopy(data())
    p = load_policy(FILES["banking"])
    step(p, "send_money", pay("A"))
    assert data() == before
