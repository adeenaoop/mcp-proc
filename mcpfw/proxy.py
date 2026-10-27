"""mcpfw: pass-through MCP stdio proxy with a hash-chained call log (guide §8).

Week 1 behaviour: forwards every byte unchanged in both directions and logs
each tools/call with its response. Blocks nothing.

Usage:
    python mcpfw/proxy.py --log run/<ep>/trace.jsonl [--tee run/<ep>/wire] -- <server command and args...>
    python -m mcpfw      --log ... -- ...          (same thing)

Standard library only: no MCP SDK, because the SDK would re-serialise
messages and change bytes.

RULE: never print to stdout. Stdout carries the protocol. Logging -> stderr.
"""
import argparse
import asyncio
import json
import sys
import time

try:                                    # python -m mcpfw / import as package
    from .events import ChainWriter
    from .wire import LINE_LIMIT, Tee, read_line
except ImportError:                     # python mcpfw/proxy.py
    from events import ChainWriter
    from wire import LINE_LIMIT, Tee, read_line


def warn(*args):
    print("[mcpfw]", *args, file=sys.stderr, flush=True)


def parse_args(argv):
    if "--" not in argv:
        sys.exit("usage: mcpfw --log FILE [--tee DIR] -- <server command and args...>")
    split = argv.index("--")
    p = argparse.ArgumentParser(prog="mcpfw")
    p.add_argument("--log", required=True, help="trace.jsonl path")
    p.add_argument("--tee", default=None, help="directory for raw byte captures (X2)")
    args = p.parse_args(argv[:split])
    args.child = argv[split + 1:]
    if not args.child:
        sys.exit("mcpfw: no server command after --")
    return args


def parse_copy(line):
    """Parse a COPY of a line for logging. Returns a dict or None. Never used for forwarding."""
    try:
        msg = json.loads(line)
    except ValueError:
        warn("unparseable line forwarded unchanged:", line[:120])
        return None
    if isinstance(msg, list):
        warn("JSON array (batch) forwarded unchanged; not logged")
        return None
    return msg if isinstance(msg, dict) else None


def result_text_and_error(msg):
    """From a response message: (result object, joined text, is_error)."""
    if "error" in msg:                              # JSON-RPC level error
        err = msg["error"]
        text = err.get("message", "") if isinstance(err, dict) else str(err)
        return err, text, True
    result = msg.get("result")
    if not isinstance(result, dict):
        return result, "", False
    text = "".join(c.get("text", "") for c in result.get("content", [])
                   if isinstance(c, dict) and c.get("type") == "text")
    return result, text, bool(result.get("isError", False))   # wire format is camelCase


class Proxy:
    def __init__(self, log, tee):
        self.log = log
        self.tee = tee
        self.pending = {}            # request id -> (seq, t_req, tool, args)
        self.seq = 0

    def observe_up(self, line):
        msg = parse_copy(line)
        if msg is None or msg.get("method") != "tools/call" or "id" not in msg:
            return
        params = msg.get("params") or {}
        key = json.dumps(msg["id"])      # keeps 1 and "1" distinct, exactly as parsed
        self.pending[key] = (self.seq, time.monotonic_ns(), params.get("name"), params.get("arguments") or {})
        self.seq += 1

    def observe_down(self, line):
        msg = parse_copy(line)
        if msg is None or "id" not in msg or ("result" not in msg and "error" not in msg):
            return
        entry = self.pending.pop(json.dumps(msg["id"]), None)
        if entry is None:                # response to some other method (initialize, tools/list, ...)
            return
        seq, t_req, tool, args = entry
        result, text, is_error = result_text_and_error(msg)
        self.log.write(seq=seq, t_req=t_req, t_resp=time.monotonic_ns(), tool=tool,
                       args=args, is_error=is_error, result_text=text, result=result)

    async def up_pump(self, stdin_reader, child):
        """Our stdin -> child's stdin."""
        try:
            while True:
                line = await read_line(stdin_reader)
                if not line:                         # §8.5 step 1: our stdin hit EOF
                    break
                self.tee.write("up_in", line)
                child.stdin.write(line)              # exact bytes read
                await child.stdin.drain()
                self.tee.write("up_out", line)
                self.observe_up(line)                # after forwarding
        except (BrokenPipeError, ConnectionResetError):
            warn("child stdin closed early")
        finally:
            try:
                child.stdin.close()                  # child sees EOF -> writes env_post.json
                await child.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    async def down_pump(self, child, stdout):
        """Child's stdout -> our stdout, until the child's stdout hits EOF."""
        while True:
            line = await read_line(child.stdout)
            if not line:                             # §8.5 step 2: drained
                break
            self.tee.write("down_in", line)
            stdout.write(line)                       # exact bytes read
            stdout.flush()
            self.tee.write("down_out", line)
            self.observe_down(line)                  # after forwarding


async def run(args):
    log = ChainWriter(args.log)
    tee = Tee(args.tee)
    proxy = Proxy(log, tee)

    loop = asyncio.get_running_loop()
    stdin_reader = asyncio.StreamReader(limit=LINE_LIMIT)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stdin_reader), sys.stdin)

    child = await asyncio.create_subprocess_exec(
        *args.child, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=None, limit=LINE_LIMIT)              # stderr passes straight through

    stdout = sys.stdout.buffer
    up = asyncio.create_task(proxy.up_pump(stdin_reader, child))
    try:
        await proxy.down_pump(child, stdout)         # returns when the child's stdout closes
        if not up.done():                            # child exited first: stop reading stdin
            up.cancel()
        try:
            await up
        except asyncio.CancelledError:
            pass
        code = await child.wait()                    # §8.5 step 3; never kill the child early
    finally:
        if proxy.pending:
            warn(f"{len(proxy.pending)} tools/call request(s) got no response; not logged")
        log.close()                                  # §8.5 step 4
        tee.close()
    return code


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    code = asyncio.run(run(args))
    sys.exit(code)                                   # §8.5 step 5: child's exit code


if __name__ == "__main__":
    main()
