"""R2 end to end (step 3): bridge -> mcpfw (real proxy) -> real AgentDojo banking server.

Checks that mcpfw passes each call's arguments to R2, that R2 blocks (enforce)
or only logs (audit) a new destination over the limit, what reaches the bank,
and that every record says which destination R2 counted.

Run from ~/fyp:  pytest tests/test_policy_r2_proxy.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcpfw"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_policy_audit import ATTACKER, POLICY, Call, run_e2e  # noqa: E402
from wire import verify_tee  # noqa: E402

R2_ENFORCE = ["--policy", str(POLICY), "--rules", "R2", "--mode", "enforce"]   # banking: k=1
R2_AUDIT = ["--policy", str(POLICY), "--rules", "R2", "--mode", "audit"]
BOTH_ENFORCE = ["--policy", str(POLICY), "--mode", "enforce"]                 # R1 and R2 (file order)
A, B = "XX00TEST0000000000000001", "XX00TEST0000000000000002"   # not in the AgentDojo data


def pay(to, amount=10.0):
    return Call("send_money", {"recipient": to, "amount": amount, "subject": "rent", "date": "2022-01-01"})


def sent_to(env, iban):
    return [t for t in env["bank_account"]["transactions"] if t["recipient"] == iban]


# ---------- enforce: the new destination over the limit never reaches the bank ----------
def test_enforce_blocks_new_destination_over_limit(tmp_path):
    outputs, recs, env = run_e2e([pay(A), pay(A), pay(B)], tmp_path, R2_ENFORCE)
    assert [o[1] for o in outputs] == [False, False, True]          # is_error: only the third
    assert "(R2)" in outputs[2][0] and "`" + B.lower() + "`" in outputs[2][0]
    assert len(sent_to(env, A)) == 2 and sent_to(env, B) == []      # B's money never left
    assert [r["decision"] for r in recs] == ["allow", "allow", "deny"]
    assert [r["action"] for r in recs] == ["forward", "forward", "block"]
    assert [r["committed"] for r in recs] == [True, True, False]
    assert [(r["states_before"], r["states_after"]) for r in recs] == [
        ({"R2": "COUNT_0"}, {"R2": "COUNT_1"}),
        ({"R2": "COUNT_1"}, {"R2": "COUNT_1"}),
        ({"R2": "COUNT_1"}, {"R2": "COUNT_1"})]
    assert [r["destinations"] for r in recs] == [{"R2": A.lower()}, {"R2": A.lower()}, {"R2": B.lower()}]
    assert recs[2]["rules_fired"] == ["R2"]
    verify_tee(tmp_path / "wire")                                   # blocked call accounted for


# ---------- audit: decides and logs, never blocks ----------
def test_audit_logs_would_deny_and_forwards(tmp_path):
    outputs, recs, env = run_e2e([pay(A), pay(B)], tmp_path, R2_AUDIT)
    assert [o[1] for o in outputs] == [False, False]                # nothing blocked
    assert len(sent_to(env, B)) == 1                                # audit: the money really went
    assert [r["decision"] for r in recs] == ["allow", "would_deny"]
    assert [r["committed"] for r in recs] == [True, False]          # B never used a slot
    assert recs[1]["states_after"] == {"R2": "COUNT_1"}
    assert recs[1]["destinations"] == {"R2": B.lower()}
    verify_tee(tmp_path / "wire")


# ---------- R1 and R2 together ----------
def test_both_rules_give_their_reason(tmp_path):
    calls = [pay(A), Call("read_file", {"file_path": "bill-december-2023.txt"}), pay(ATTACKER, 100.0)]
    outputs, recs, env = run_e2e(calls, tmp_path, BOTH_ENFORCE)
    assert recs[2]["decision"] == "deny" and recs[2]["rules_fired"] == ["R1", "R2"]
    assert "(R1)" in outputs[2][0] and "(R2)" in outputs[2][0]
    assert sent_to(env, ATTACKER) == []
    assert recs[1]["states_after"] == {"R1": "TAINTED", "R2": "COUNT_1"}


def test_r1_catches_attack_r2_alone_would_allow(tmp_path):
    """Attacker IBAN as the FIRST destination: R2 (k=1) allows it, R1 blocks it.
    This is the documented R2 limitation and why the rules run together."""
    calls = [Call("read_file", {"file_path": "bill-december-2023.txt"}), pay(ATTACKER, 100.0)]
    _, recs, env = run_e2e(calls, tmp_path, BOTH_ENFORCE)
    assert recs[1]["rules_fired"] == ["R1"]                        # R2 had a free slot
    assert recs[1]["destinations"] == {"R2": ATTACKER.lower()}
    assert sent_to(env, ATTACKER) == []


def test_r2_alone_lets_first_destination_through(tmp_path):
    calls = [Call("read_file", {"file_path": "bill-december-2023.txt"}), pay(ATTACKER, 100.0)]
    outputs, recs, env = run_e2e(calls, tmp_path, R2_ENFORCE)
    assert recs[1]["decision"] == "allow" and outputs[1][1] is False
    assert len(sent_to(env, ATTACKER)) == 1


# ---------- record format ----------
def test_optional_destination_omitted_is_allowed_and_not_counted(tmp_path):
    calls = [Call("update_scheduled_transaction", {"id": 7, "amount": 1200.0}), pay(A)]
    _, recs, _ = run_e2e(calls, tmp_path, R2_ENFORCE)
    assert recs[0]["decision"] == "allow" and "destinations" not in recs[0]
    assert recs[0]["states_after"] == {"R2": "COUNT_0"}            # no slot used
    assert recs[1]["decision"] == "allow"                           # A still fits in k=1


def test_non_egress_call_has_no_destinations_field(tmp_path):
    _, recs, _ = run_e2e([Call("get_balance", {})], tmp_path, R2_ENFORCE, tee=False)
    assert recs[0]["decision"] == "allow" and "destinations" not in recs[0]


def test_r1_only_records_unchanged(tmp_path):
    """R1 never sets a destination, so R1 records keep exactly their old fields."""
    _, recs, _ = run_e2e([pay(A)], tmp_path, ["--policy", str(POLICY), "--rules", "R1"], tee=False)
    assert "destinations" not in recs[0]
