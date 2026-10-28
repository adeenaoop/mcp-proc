"""Audit-mode would-deny observations on ground-truth replays (step 3).

For every user task of a suite, replays the task's GROUND-TRUTH tool calls
(the correct, benign solution) through bridge -> mcpfw (policy, AUDIT mode)
-> server, then reads the hash-chained trace and counts the calls the policy
WOULD have denied.

What this is:   which correct benign calls R1 would block, on these exact traces.
What this is NOT: a false-positive rate (a real agent may take other paths and
                  reacts to blocks) and NOT evidence of attack prevention
                  (no attacks are run here).

Usage (from ~/fyp, venv active):
    python harness/audit_report.py                        # banking, policies/banking.json, all rules
    python harness/audit_report.py --rules R1
Writes run/audit/<suite>_<rules>_audit.md and prints it.
"""
import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "mcpfw"))

from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from bridge import MCPBridge  # noqa: E402
from events import verify_chain  # noqa: E402
from policy import load_policy  # noqa: E402
from run_direct import numeric_task_ids  # noqa: E402

SERVER = ROOT / "server" / "agentdojo_mcp_server.py"
PROXY = ROOT / "mcpfw" / "proxy.py"


def replay(suite, suite_name, bench, task_id, policy_path, rules, d):
    task = suite.user_tasks[task_id]
    calls = task.ground_truth(task.init_environment(suite.load_and_inject_default_environment({})))
    server = [str(SERVER), "--suite", suite_name, "--bench-version", bench, "--user-task", task_id,
              "--env-pre", str(d / "env_pre.json"), "--env-out", str(d / "env_post.json")]
    opts = ["--policy", str(policy_path)] + (["--rules", rules] if rules else [])
    bridge = MCPBridge(sys.executable,
                       [str(PROXY), "--log", str(d / "trace.jsonl"), *opts, "--", sys.executable, *server],
                       expected_tools={f.name for f in suite.tools})
    try:
        for c in calls:
            bridge.call(c.function, dict(c.args))
    finally:
        bridge.close()
    records = verify_chain(d / "trace.jsonl")
    assert len(records) == len(calls), "trace does not have one record per call"
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--bench", default="v1.2.2")
    ap.add_argument("--policy", default=str(ROOT / "policies" / "banking.json"))
    ap.add_argument("--rules", default=None, help="comma-separated rule ids (default: all)")
    args = ap.parse_args()

    suite = get_suite(args.bench, args.suite)
    policy = load_policy(args.policy, None if args.rules is None else args.rules.split(","))

    rows, n_calls, n_would_deny, tasks_hit, errors = [], 0, 0, 0, 0
    for tid in numeric_task_ids(suite):
        with tempfile.TemporaryDirectory() as tmp:
            recs = replay(suite, args.suite, args.bench, tid, args.policy, args.rules, Path(tmp))
        denied = [r for r in recs if r["decision"] == "would_deny"]
        errors += sum(r["decision"] == "policy_error" for r in recs)
        n_calls += len(recs)
        n_would_deny += len(denied)
        tasks_hit += bool(denied)
        seq = " → ".join(f"**{r['tool']}** ✗" if r["decision"] == "would_deny" else r["tool"] for r in recs)
        fired = ", ".join(sorted({x for r in denied for x in r["rules_fired"]})) or "–"
        rows.append(f"| {tid} | {len(recs)} | {len(denied)} | {fired} | {seq} |")

    rules_label = "+".join(policy.enabled)
    md = "\n".join([
        f"# Audit-mode would-deny observations on ground-truth replays",
        "",
        f"- Suite: **{args.suite}**, AgentDojo benchmark **{args.bench}**",
        f"- Policy: `{Path(args.policy).name}`, fingerprint `{policy.fingerprint[:16]}`, rules **{rules_label}**",
        f"- Mode: **audit** (decided and logged, nothing blocked). Generated {date.today().isoformat()}.",
        "",
        "**How to read this.** Each row replays the task's *ground-truth* calls, i.e. the correct,",
        "benign solution. A call marked ✗ is one the policy *would* deny on that exact trace.",
        "This is **not a false-positive rate**: a real agent may take a different path, and in",
        "enforce mode it would react to a block. It is **not evidence of attack prevention**:",
        "no attacks are run here.",
        "",
        f"**Summary:** {n_would_deny} of {n_calls} benign ground-truth calls would be denied; "
        f"{tasks_hit} of {len(rows)} tasks have at least one would-deny call. "
        f"Policy errors: {errors}.",
        "",
        "| Task | Calls | Would deny | Rules | Ground-truth sequence (✗ = would deny) |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ])
    out = ROOT / "run" / "audit" / f"{args.suite}_{rules_label}_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"written: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
