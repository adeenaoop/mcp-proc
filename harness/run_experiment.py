"""R1 experiment: AgentDojo episodes in three configurations, plus a results table.

Configurations
  A  direct      AgentDojo as published: tools are plain Python calls. No MCP, no defense.
  B  mcp         agent -> MCPRuntime -> bridge -> mcpfw (no policy) -> MCP server. No defense.
  C  mcp+R1      same as B, but mcpfw enforces R1 (--mode enforce).
  C_audit        same as C, but --mode audit: R1 decides and logs, never blocks.
                 Should match B exactly; its log shows what R1 WOULD block (would_deny).

Every MCP episode (B, C, C_audit) also records: hash-chain check, wire fidelity
(raw byte captures checked with verify_tee), peak memory of mcpfw, and per-call
policy decisions.

Episodes per suite and seed (exactly AgentDojo's pairing):
  benign    every user task, no injection                 (utility)
  attacked  every user task x every injection task          (utility under attack, ASR)
            injections built by AgentDojo's attack (default: tool_knowledge)

Usage (from ~/fyp, venv active):
  python harness/run_experiment.py --fake                    # dry run, no key, ~5 min: checks everything works
  python harness/run_experiment.py                           # REAL run (needs LLM_API_KEY in .env)
  python harness/run_experiment.py --suites banking --configs B,C --limit 3   # small pilot
  python harness/run_experiment.py --summarize-only --name r1_main            # rebuild the table

Resumable: an episode with a result.json is skipped, so after a crash or a
Ctrl+C just run the same command again. Failed episodes are listed in
errors.jsonl and retried on the next run.

Outputs in run/exp/<name>/:
  <config>/<suite>_<user>_<inj>_r<seed>/result.json   one per episode (+ trace.jsonl etc. for B, C)
  summary.md, summary.csv, manifest.json, errors.jsonl

--fake uses fake models (no LLM): benign episodes replay the ground truth;
attacked episodes use a fully GULLIBLE fake that obeys every injection it can
see. Its numbers only prove the pipeline works. They are NOT results.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "mcpfw"))

from agentdojo.attacks import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from counting_client import CountingClient, CountingRuntime  # noqa: E402
from fake_llm import GroundTruthFakeClient, gullible_calls  # noqa: E402
from policy import load_policy  # noqa: E402
from run_direct import build_pipeline, make_client, numeric_task_ids  # noqa: E402
from run_episode import run_episode  # noqa: E402

CONFIGS = {
    "A": "direct AgentDojo (no MCP, no defense)",
    "B": "MCP via mcpfw, no policy (no defense)",
    "C": "MCP via mcpfw, R1 enforced",
    "C_audit": "MCP via mcpfw, R1 audit only (decides and logs, never blocks)",
}
POLICY_CONFIGS = {"C", "C_audit"}
FAKE_ATTACK_MODEL = "gpt-4o-mini-2024-07-18"   # attacks need a known model name, even for fakes


def log(*a):
    print(*a, flush=True)


def policy_path(suite_name):
    return ROOT / "policies" / f"{suite_name}.json"


def proxy_options(config, suite_name):
    if config == "C":
        return ["--policy", str(policy_path(suite_name)), "--rules", "R1", "--mode", "enforce"]
    if config == "C_audit":
        return ["--policy", str(policy_path(suite_name)), "--rules", "R1", "--mode", "audit"]
    return []


# ---------------- jobs ----------------
def build_jobs(args, suites):
    """All episodes, in a fixed order. Injections are built here, in one thread."""
    jobs = []
    for suite_name, suite in suites.items():
        attack_pipeline = build_pipeline(None, FAKE_ATTACK_MODEL if args.fake else args.model_name)
        attack = load_attack(args.attack, suite, attack_pipeline)
        user_ids = numeric_task_ids(suite)[: args.limit] if args.limit else numeric_task_ids(suite)
        inj_ids = sorted(suite.injection_tasks, key=lambda k: int(k.rsplit("_", 1)[-1]))
        pairs = [(u, None, {}) for u in user_ids]
        for u in user_ids:
            for i in inj_ids:
                pairs.append((u, i, attack.attack(suite.user_tasks[u], suite.injection_tasks[i])))
        for config in args.configs:
            for seed in range(args.seeds):
                for u, i, injections in pairs:
                    jobs.append(dict(suite=suite_name, config=config, seed=seed, user=u, inj=i,
                                     injections=injections))
    return jobs


def ep_dir(exp_dir, job):
    return exp_dir / job["config"] / f"{job['suite']}_{job['user']}_{job['inj'] or 'none'}_r{job['seed']}"


# ---------------- clients ----------------
def fake_client(suite, job, sources):
    user_task = suite.user_tasks[job["user"]]
    env = user_task.init_environment(suite.load_and_inject_default_environment({}))
    calls = user_task.ground_truth(env)
    if job["inj"] is not None:
        inj_calls = suite.injection_tasks[job["inj"]].ground_truth(suite.load_and_inject_default_environment({}))
        calls = gullible_calls(calls, inj_calls, sources)
    return GroundTruthFakeClient(calls, user_task.GROUND_TRUTH_OUTPUT)


# ---------------- one episode ----------------
def run_direct_episode(suite, job, client, model, bench, d):
    """Config A: AgentDojo's own run_task_with_pipeline (tools as Python calls)."""
    # A fresh subclass per episode: CountingRuntime counts on the class, so
    # parallel episodes must not share it.
    runtime_cls = type("EpisodeRuntime", (CountingRuntime,), {"n_tool_calls": 0})
    pipeline = build_pipeline(client, model)
    client.reset()
    t0 = time.monotonic()
    utility, security = suite.run_task_with_pipeline(
        pipeline, suite.user_tasks[job["user"]],
        injection_task=suite.injection_tasks[job["inj"]] if job["inj"] else None,
        injections=job["injections"], runtime_class=runtime_cls)
    record = {"mode": "direct", "suite": job["suite"], "bench_version": bench, "model": model,
              "task": job["user"], "injection_task": job["inj"], "rep": job["seed"],
              "utility": bool(utility), "security": bool(security),
              "llm_calls": client.llm_calls, "tok_in": client.tok_in, "tok_out": client.tok_out,
              "n_tool_calls": runtime_cls.n_tool_calls, "wall_s": round(time.monotonic() - t0, 2)}
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(record, indent=2))
    return record


def run_one(job, suites, args, real_client, exp_dir, sources):
    suite = suites[job["suite"]]
    d = ep_dir(exp_dir, job)
    if args.fake:
        client, model = CountingClient(fake_client(suite, job, sources[job["suite"]])), "fake"
    else:
        client, model = CountingClient(real_client), args.model_name
    if job["config"] == "A":
        record = run_direct_episode(suite, job, client, model, args.bench, d)
    else:
        record = run_episode(suite, job["suite"], args.bench, job["user"], job["seed"], client, model,
                             exp_dir / job["config"], injection_task_id=job["inj"],
                             injections=job["injections"], via_proxy=True, force=True,
                             proxy_extra=proxy_options(job["config"], job["suite"]), measure=True)
    record["config"] = job["config"]
    record["fake"] = bool(args.fake)
    (d / "result.json").write_text(json.dumps(record, indent=2))
    return record


# ---------------- summary ----------------
def rate(xs):
    return f"{100 * sum(xs) / len(xs):.1f}%" if xs else "–"


def mean(xs, scale=1.0, fmt="{:.1f}"):
    return fmt.format(statistics.mean(xs) * scale) if xs else "–"


def count_ok(rs, key):
    """'n_true/n_with_key', or '–' if no episode recorded the key."""
    xs = [r[key] for r in rs if key in r]
    return f"{sum(bool(x) for x in xs)}/{len(xs)}" if xs else "–"


def kb_to_mb(xs, how):
    xs = [x for x in xs if x is not None]
    return f"{how(xs) / 1024:.1f}" if xs else "–"


def summarize(exp_dir):
    manifest = json.loads((exp_dir / "manifest.json").read_text()) if (exp_dir / "manifest.json").exists() else {}
    results = [json.loads(p.read_text()) for p in sorted(exp_dir.glob("*/*/result.json"))]
    errors = []
    if (exp_dir / "errors.jsonl").exists():
        errors = [json.loads(l) for l in (exp_dir / "errors.jsonl").read_text().splitlines() if l.strip()]
    fake = any(r.get("fake") for r in results)

    rows = []
    for suite in sorted({r["suite"] for r in results}):
        for config in sorted({r["config"] for r in results}):
            rs = [r for r in results if r["suite"] == suite and r["config"] == config]
            if not rs:
                continue
            benign = [r for r in rs if not r["injection_task"]]
            attacked = [r for r in rs if r["injection_task"]]
            calls = [r for r in rs if "call_latency_ns" in r]
            policy = config in POLICY_CONFIGS
            # call-level FPR: in a BENIGN episode every call is legitimate, so every
            # blocked (C) or would-be-blocked (C_audit) call there is a false positive
            legit_calls = sum(r["n_tool_calls"] for r in benign)
            fp_calls = sum(r.get("n_blocked", 0) + r.get("n_would_deny", 0) for r in benign)
            dec = {}
            for r in rs:
                for k, v in r.get("decisions", {}).items():
                    dec[k] = dec.get(k, 0) + v
            rows.append({
                "suite": suite, "config": config, "description": CONFIGS.get(config, ""),
                "benign_n": len(benign), "benign_utility": rate([r["utility"] for r in benign]),
                "attacked_n": len(attacked), "utility_under_attack": rate([r["utility"] for r in attacked]),
                "ASR": rate([r["security"] for r in attacked]),
                "benign_eps_with_block": (f"{sum(r.get('n_blocked', 0) > 0 for r in benign)}/{len(benign)}"
                                          if config == "C" else "–"),
                "attacked_eps_with_block": (f"{sum(r.get('n_blocked', 0) > 0 for r in attacked)}/{len(attacked)}"
                                            if config == "C" else "–"),
                "blocked_calls": sum(r.get("n_blocked", 0) for r in rs) if config == "C" else "–",
                "benign_calls": legit_calls,
                "benign_calls_flagged": fp_calls if policy else "–",
                "FPR_calls": (f"{100 * fp_calls / legit_calls:.1f}%" if policy and legit_calls else "–"),
                "dec_allow": dec.get("allow", 0) if policy else "–",
                "dec_would_deny": dec.get("would_deny", 0) if policy else "–",
                "dec_deny": dec.get("deny", 0) if policy else "–",
                "dec_policy_error": dec.get("policy_error", 0) if policy else "–",
                "chain_ok": count_ok(rs, "chain_ok"),
                "wire_ok": count_ok(rs, "wire_ok"),
                "proxy_rss_mb_mean": kb_to_mb([r.get("proxy_peak_rss_kb") for r in rs], statistics.mean),
                "proxy_rss_mb_max": kb_to_mb([r.get("proxy_peak_rss_kb") for r in rs], max),
                "policy_errors": sum(r.get("policy_errors", 0) for r in rs),
                "trace_mismatch": sum(1 for r in rs if r.get("trace_matches_tool_calls") is False),
                "tok_in_mean": mean([r["tok_in"] for r in rs], fmt="{:.0f}"),
                "tok_out_mean": mean([r["tok_out"] for r in rs], fmt="{:.0f}"),
                "llm_calls_mean": mean([r["llm_calls"] for r in rs]),
                "wall_s_mean": mean([r["wall_s"] for r in rs], fmt="{:.2f}"),
                "call_latency_ms_mean": mean([x for r in calls for x in r["call_latency_ns"]], 1e-6, "{:.2f}"),
                "decide_us_mean": mean([x for r in rs for x in r.get("decide_ns", [])], 1e-3, "{:.1f}"),
                "errors": sum(1 for e in errors if e["suite"] == suite and e["config"] == config),
            })

    with open(exp_dir / "summary.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    banner = []
    if fake:
        banner = ["> **SIMULATED DRY RUN (fake models, no LLM). These numbers only show the pipeline works.**",
                  "> Benign = ground-truth replay; attacked = a fully gullible fake that obeys every visible",
                  "> injection. They are NOT experimental results.", ""]
    cols = ["suite", "config", "benign_n", "benign_utility", "attacked_n", "utility_under_attack", "ASR",
            "benign_eps_with_block", "attacked_eps_with_block", "blocked_calls", "policy_errors",
            "trace_mismatch", "tok_in_mean", "wall_s_mean", "call_latency_ms_mean", "decide_us_mean", "errors"]
    cols2 = ["suite", "config", "benign_calls", "benign_calls_flagged", "FPR_calls", "dec_allow",
             "dec_would_deny", "dec_deny", "dec_policy_error", "chain_ok", "wire_ok",
             "proxy_rss_mb_mean", "proxy_rss_mb_max"]
    md = [f"# R1 experiment: {exp_dir.name}", "", *banner,
          f"- Model: `{manifest.get('model', '?')}`, attack: `{manifest.get('attack', '?')}`, "
          f"seeds: {manifest.get('seeds', '?')}, AgentDojo `{manifest.get('bench', '?')}`",
          f"- Policy fingerprints: {manifest.get('policy_fingerprints', {})}",
          f"- Episodes with results: {len(results)}; errors: {len(errors)}", "",
          "**Columns.** Utility = AgentDojo utility check passed. ASR = attack success rate "
          "(AgentDojo security check: the injection goal was achieved; lower is better). "
          "`benign_eps_with_block` = benign episodes where R1 blocked at least one call "
          "(the false-positive signal). Latency = per forwarded tool call through mcpfw. "
          "`decide_us` = time for the policy decision.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md += ["", "**Integrity, decisions, memory.** `FPR_calls` = flagged calls in benign episodes / all "
           "calls in benign episodes (every benign call is legitimate; C counts blocked calls, "
           "C_audit counts would-be-blocked calls). `dec_*` = per-call policy decisions from "
           "trace.jsonl. `chain_ok` = episodes whose hash chain verified. `wire_ok` = episodes whose "
           "raw byte captures verified (every byte forwarded or blocked, in order). Memory = peak "
           "resident set size of the mcpfw process in MB (Linux VmHWM, read by mcpfw at exit).", "",
           "| " + " | ".join(cols2) + " |", "|" + "---|" * len(cols2)]
    md += ["| " + " | ".join(str(r[c]) for c in cols2) + " |" for r in rows]
    md += ["", "Configs: " + "; ".join(f"**{k}** = {v}" for k, v in CONFIGS.items()), ""]
    (exp_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="experiment folder name (default: r1_main or r1_fake)")
    ap.add_argument("--suites", default="banking,slack")
    ap.add_argument("--configs", default="A,B,C,C_audit")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N user tasks only (pilot)")
    ap.add_argument("--attack", default="tool_knowledge")
    ap.add_argument("--bench", default="v1.2.2")
    ap.add_argument("--workers", type=int, default=4, help="episodes in parallel")
    ap.add_argument("--fake", action="store_true", help="dry run with fake models (no key, not results)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in args.configs if c not in CONFIGS]
    if bad:
        sys.exit(f"unknown configs {bad}; choose from {list(CONFIGS)}")
    name = args.name or ("r1_fake" if args.fake else "r1_main")
    exp_dir = ROOT / "run" / "exp" / name
    if args.summarize_only:
        print(summarize(exp_dir))
        return

    real_client = None
    if args.fake:
        args.model_name = "fake"
    else:
        real_client, args.model_name = make_client(fake=False)

    suite_names = [s.strip() for s in args.suites.split(",") if s.strip()]
    suites = {n: get_suite(args.bench, n) for n in suite_names}
    for n in suite_names:
        if not policy_path(n).exists():
            sys.exit(f"missing policy file {policy_path(n)}")
    policies = {n: load_policy(policy_path(n), ["R1"]) for n in suite_names}
    sources = {n: set(policies[n].rules["R1"].sources) for n in suite_names}

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "manifest.json").write_text(json.dumps({
        "started": datetime.now(timezone.utc).isoformat(), "model": args.model_name, "fake": args.fake,
        "attack": args.attack, "bench": args.bench, "seeds": args.seeds, "suites": suite_names,
        "configs": {c: CONFIGS[c] for c in args.configs}, "limit": args.limit,
        "policy_fingerprints": {n: p.fingerprint[:16] for n, p in policies.items()},
    }, indent=2))

    log("building episode list (AgentDojo attack injections)...")
    jobs = build_jobs(args, suites)
    todo = [j for j in jobs if not (ep_dir(exp_dir, j) / "result.json").exists()]
    log(f"{len(jobs)} episodes in total, {len(jobs) - len(todo)} already done, {len(todo)} to run, "
        f"{args.workers} in parallel" + ("  [FAKE dry run]" if args.fake else ""))

    err_path = exp_dir / "errors.jsonl"
    if err_path.exists():
        err_path.unlink()                         # failures are retried; old ones are cleared
    err_lock = threading.Lock()
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j, suites, args, real_client, exp_dir, sources): j for j in todo}
        try:
            for fut in as_completed(futures):
                job = futures[fut]
                done += 1
                tag = f"{job['config']} {job['suite']} {job['user']} {job['inj'] or 'benign'} r{job['seed']}"
                try:
                    r = fut.result()
                    extra = (f" blocked={r.get('n_blocked', 0)}" if job["config"] == "C" else
                             f" would_deny={r.get('n_would_deny', 0)}" if job["config"] == "C_audit" else "")
                    sec = f" attack_success={r['security']}" if job["inj"] else ""
                    log(f"[{done}/{len(todo)}] {tag}: utility={r['utility']}{sec}{extra}")
                except Exception as e:                # noqa: BLE001: record and keep going
                    log(f"[{done}/{len(todo)}] {tag}: ERROR {type(e).__name__}: {e}")
                    with err_lock, open(err_path, "a") as f:
                        f.write(json.dumps({**{k: job[k] for k in ("suite", "config", "seed", "user", "inj")},
                                            "error": f"{type(e).__name__}: {e}",
                                            "trace": traceback.format_exc()[-2000:]}) + "\n")
        except KeyboardInterrupt:
            log("interrupted: finished episodes are kept; rerun the same command to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    log(f"finished in {time.monotonic() - t0:.0f}s")
    print(summarize(exp_dir))
    log(f"\nwritten: {exp_dir / 'summary.md'} and summary.csv")


if __name__ == "__main__":
    main()
