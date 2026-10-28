"""The experiment runner works end to end (fake models, no key, ~1 min).

Checks: every episode finishes, no errors, configs A and B give identical
results (MCP wrapping changes nothing), R1 (config C) blocks the simulated
egress attacks, traces match, and a rerun resumes instead of repeating work.

Run from ~/fyp:  pytest tests/test_experiment.py -v
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "pytest_fake"
EXP = ROOT / "run" / "exp" / NAME


def run(*extra):
    return subprocess.run([sys.executable, str(ROOT / "harness" / "run_experiment.py"), "--fake",
                           "--name", NAME, "--suites", "banking", "--limit", "1", "--workers", "6", *extra],
                          capture_output=True, text=True, timeout=600, cwd=ROOT)


def test_fake_experiment_end_to_end():
    shutil.rmtree(EXP, ignore_errors=True)
    try:
        p = run()
        assert p.returncode == 0, p.stderr[-2000:]
        results = [json.loads(f.read_text()) for f in EXP.glob("*/*/result.json")]
        assert len(results) == 4 * (1 + 9)                     # A,B,C,C_audit x (1 benign + 9 attacked)
        assert not (EXP / "errors.jsonl").exists() or not (EXP / "errors.jsonl").read_text().strip()

        by = {(r["config"], r["injection_task"]): r for r in results}
        # A and B: identical outcomes, episode by episode
        for (cfg, inj), r in by.items():
            if cfg == "A":
                b = by[("B", inj)]
                assert (r["utility"], r["security"]) == (b["utility"], b["security"]), inj
        # every MCP trace has one record per tool call
        assert all(r["trace_matches_tool_calls"] for r in results if r["config"] in "BC")
        # R1 blocks every simulated egress attack the gullible fake actually attempts
        egress_attacks = [f"injection_task_{i}" for i in (0, 1, 2, 3, 4, 5, 6, 8)]
        for inj in egress_attacks:
            if by[("B", inj)]["security"]:                     # attack succeeded without defense
                assert by[("C", inj)]["security"] is False, f"R1 did not stop {inj}"
                assert by[("C", inj)]["n_blocked"] >= 1

        # C_audit never blocks: same outcomes as B, but logs what R1 would deny where C denied
        for inj in [None] + [f"injection_task_{i}" for i in range(9)]:
            b, a, c = by[("B", inj)], by[("C_audit", inj)], by[("C", inj)]
            assert (a["utility"], a["security"]) == (b["utility"], b["security"]), inj
            assert a["n_blocked"] == 0 and a["decisions"].get("deny", 0) == 0
            assert a["n_would_deny"] >= 1 if c["n_blocked"] else True, inj
        # every MCP episode: hash chain and wire captures verified, memory measured
        mcp = [r for r in results if r["config"] != "A"]
        assert all(r["chain_ok"] for r in mcp)
        assert all(r["wire_ok"] for r in mcp), [r.get("wire_error") for r in mcp if not r["wire_ok"]]
        assert all(r["proxy_peak_rss_kb"] > 0 and r["rss_source"] == "VmHWM" for r in mcp)
        # mcpfw is standard-library only: its OWN peak must be far below the ~200 MB harness
        assert all(r["proxy_peak_rss_kb"] < 100 * 1024 for r in mcp)
        assert all(sum(r["decisions"].values()) == r["trace_records"] for r in mcp)
        assert all(set(r["decisions"]) <= {"none"} for r in mcp if r["config"] == "B")
        # enforce: every blocked call was a deny decision
        assert all(r["decisions"].get("deny", 0) == r["n_blocked"] for r in mcp if r["config"] == "C")

        rows = list(csv.DictReader(open(EXP / "summary.csv")))
        assert {r["config"] for r in rows} == {"A", "B", "C", "C_audit"}
        row = {r["config"]: r for r in rows}
        assert row["C"]["FPR_calls"].endswith("%") and row["B"]["FPR_calls"] == "–"
        assert row["C"]["wire_ok"] == row["C"]["chain_ok"] == "10/10"
        assert "SIMULATED" in (EXP / "summary.md").read_text()

        # resume: a second run finds everything done
        p2 = run()
        assert "0 to run" in p2.stdout
    finally:
        shutil.rmtree(EXP, ignore_errors=True)
