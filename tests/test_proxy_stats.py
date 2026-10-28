"""mcpfw --stats and orderly SIGTERM shutdown.

Why: the MCP client SDK closes stdin, waits 2 s, then SIGTERMs the server
process tree. Under load mcpfw can still be shutting down then. It must close
its log and captures and still write --stats, with its OWN memory (not the
parent's, which ru_maxrss would report).

Run from ~/fyp:  pytest tests/test_proxy_stats.py -v
"""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcpfw"))
from events import verify_chain  # noqa: E402
from wire import verify_tee  # noqa: E402

PROXY = ROOT / "mcpfw" / "proxy.py"


def start(tmp_path, server):
    return subprocess.Popen([sys.executable, str(PROXY), "--log", str(tmp_path / "t.jsonl"),
                             "--tee", str(tmp_path / "w"), "--stats", str(tmp_path / "s.json"),
                             "--", *server], stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def test_stats_on_normal_exit(tmp_path):
    p = start(tmp_path, [sys.executable, "-c", "import sys; sys.stdin.read()"])
    p.stdin.close()
    assert p.wait(timeout=20) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["exit_code"] == 0 and s["rss_source"] == "VmHWM"
    assert 0 < s["proxy_peak_rss_kb"] < 100 * 1024          # own memory, standard library only


def test_sigterm_closes_log_and_writes_stats(tmp_path):
    server = subprocess.Popen(["sleep", "0"])                 # warm-up: make sure `sleep` exists
    server.wait()
    p = start(tmp_path, ["sleep", "60"])                      # a server that never exits by itself
    time.sleep(1.0)
    p.stdin.close()
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 143                          # 128 + SIGTERM
    assert b"stopped by SIGTERM" in p.stderr.read()
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["exit_code"] == 143 and s["proxy_peak_rss_kb"] > 0
    verify_chain(tmp_path / "t.jsonl")                        # log closed cleanly
    verify_tee(tmp_path / "w")                                # captures consistent


def test_no_stats_file_without_flag(tmp_path):
    p = subprocess.Popen([sys.executable, str(PROXY), "--log", str(tmp_path / "t.jsonl"), "--",
                          sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    p.stdin.close()
    assert p.wait(timeout=20) == 0
    assert not list(tmp_path.glob("*.json"))
