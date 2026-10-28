"""mcpfw: MCP stdio proxy with a hash-chained call log (guide §8).

Without --policy: pure pass-through. Forwards every byte unchanged in both
directions and logs each tools/call with its response. This is exactly the
Week 1 code path.

With --policy (step 3, AUDIT mode): before forwarding each tools/call, the
policy is asked for a decision, and the decision is written into that call's
log record. Audit mode NEVER blocks: every call is forwarded unchanged. A call
the policy would deny is logged as "would_deny" and does NOT change policy
state (as if it had been blocked), so the log shows what enforcement would
decide on this exact trace. Enforcement (actually blocking) is step 4.

Lifecycle per tools/call:
    parse a copy -> [decide] -> register as pending -> forward ORIGINAL bytes
    -> [commit] -> mark dispatched
The call is registered BEFORE its bytes are sent, so a fast server's response
can never arrive for a call the proxy does not know about. Its log record is
written only when BOTH the response has arrived AND dispatch has finished
(commit outcome known); a response that comes first is held until then.

Commit criterion = successful DISPATCH: the request bytes were written and
drain() returned. This does NOT mean the tool ran successfully; drain() only
says the bytes left the proxy's buffer. If writing fails, the call is dropped
from pending and never committed.

Not covered yet (step 4/6): JSON arrays (batches) and tools/call messages
without an id are forwarded WITHOUT a policy decision (a warning is printed).
Audit coverage is therefore "tools/call requests with an id", not every
possible message.

Malformed calls (params not an object, missing/invalid tool name) never crash
the proxy: they are forwarded unchanged and logged with decision
"policy_error" (no real decision made), never "allow". Two in-flight calls
with the same id are both kept (FIFO per id) and both records are marked
"duplicate_id": true; step 4 will deny duplicates.

Usage:
    python mcpfw/proxy.py --log run/<ep>/trace.jsonl [--tee DIR] \
        [--policy policies/banking.json [--rules R1,R2] [--mode audit]] \
        -- <server command and args...>

Exit codes: the server's exit code; 2 if the policy file or --rules is invalid
(checked BEFORE the server is started, so no server ever runs unprotected by
mistake).

Standard library only. RULE: never print to stdout (it carries the protocol).
"""
import argparse
import asyncio
import json
import sys
import time

try:                                    # python -m mcpfw / import as package
    from .events import ChainWriter
    from .policy import PolicyError, load_policy
    from .wire import LINE_LIMIT, Tee, read_line
except ImportError:                     # python mcpfw/proxy.py
    from events import ChainWriter
    from policy import PolicyError, load_policy
    from wire import LINE_LIMIT, Tee, read_line

EXIT_BAD_POLICY = 2


def warn(*args):
    print("[mcpfw]", *args, file=sys.stderr, flush=True)


def parse_args(argv):
    if "--" not in argv:
        sys.exit("usage: mcpfw --log FILE [--tee DIR] [--policy FILE [--rules R1,..] [--mode audit]] -- <server...>")
    split = argv.index("--")
    p = argparse.ArgumentParser(prog="mcpfw")
    p.add_argument("--log", required=True, help="trace.jsonl path")
    p.add_argument("--tee", default=None, help="directory for raw byte captures (X2)")
    p.add_argument("--policy", default=None, help="policy JSON file; omit for pure pass-through")
    p.add_argument("--rules", default=None, help="comma-separated rule ids to enable (default: all in file)")
    p.add_argument("--mode", default="audit", choices=["audit"],
                   help="audit = decide and log, never block (enforce comes in step 4)")
    args = p.parse_args(argv[:split])
    if args.rules is not None and args.policy is None:
        p.error("--rules needs --policy")
    args.child = argv[split + 1:]
    if not args.child:
        sys.exit("mcpfw: no server command after --")
    return args


def parse_copy(line):
    """Parse a COPY of a line. Returns a dict or None. Never used for forwarding."""
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


def id_key(msg_id):
    return json.dumps(msg_id)            # keeps 1 and "1" distinct, exactly as parsed


def split_call(msg):
    """(tool, args, problem) from a tools/call message. Never raises.

    problem is None for a well-formed call, otherwise a short description
    (malformed params, invalid tool name). Malformed calls are still forwarded
    unchanged and logged; they just get no real policy decision.
    """
    params = msg.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None, None, f"malformed params: expected an object, got {type(params).__name__}"
    tool = params.get("name")
    args = params.get("arguments")
    if args is None:
        args = {}
    if not isinstance(tool, str) or not tool:
        return tool, args, f"invalid tool name {tool!r}"
    return tool, args, None


class _Pending:
    """One tools/call in flight."""
    __slots__ = ("key", "seq", "t_req", "tool", "args", "extra", "dispatched", "response", "duplicate")

    def __init__(self, key, seq, t_req, tool, args, extra):
        self.key, self.seq, self.t_req = key, seq, t_req
        self.tool, self.args, self.extra = tool, args, extra
        self.dispatched = False          # request bytes sent (and commit attempted)?
        self.response = None             # (t_resp, result, text, is_error) if it came first
        self.duplicate = False           # another in-flight call had the same id


class Proxy:
    def __init__(self, log, tee, policy=None, mode="audit"):
        self.log = log
        self.tee = tee
        self.policy = policy
        self.mode = mode
        # request id key -> FIFO list of _Pending. A list, not a single entry,
        # so two in-flight calls with the same id never overwrite each other.
        self.pending = {}
        self.seq = 0
        self.list_ids = set()            # ids of tools/list requests (for the name check)
        self.listed_tools = set()

    def pending_count(self):
        return sum(len(q) for q in self.pending.values())

    # ---------------- policy (audit) ----------------
    def check(self, tool, problem=None):
        """Ask the policy about one call. Returns (PolicyDecision or None, log fields).

        decision values:
            allow         policy allows; committed after successful dispatch
            would_deny    policy would block; forwarded anyway (audit), NOT committed
            policy_error  no real decision was made (malformed call, or the policy
                          failed); forwarded (audit), NOT committed, never reported as allow
        """
        states = self.policy.states()
        fields = {"policy_fp": self.policy.fingerprint, "mode": self.mode,
                  "rules_enabled": list(self.policy.enabled), "committed": False}
        t0 = time.monotonic_ns()
        try:
            if problem is not None:
                raise ValueError(problem)
            pd = self.policy.decide(tool)
        except Exception as e:                       # noqa: BLE001 (any policy failure)
            fields.update(decision="policy_error", rules_fired=[], states_before=states,
                          states_after=states, reason="", policy_error=f"{type(e).__name__}: {e}",
                          decide_ns=time.monotonic_ns() - t0)
            warn(f"no policy decision for tool {tool!r}: {e} (audit mode: forwarded, not committed)")
            return None, fields
        fields.update(decision="allow" if pd.allow else "would_deny",
                      rules_fired=list(pd.rules_fired), states_before=pd.states_before(),
                      states_after=pd.states_after(), reason=pd.reason,
                      decide_ns=time.monotonic_ns() - t0)
        return pd, fields

    def commit(self, pd, fields):
        """Called only after successful dispatch. Afterwards states_after is set
        to the policy's ACTUAL state (not the predicted one), so a failed commit
        can never be logged as a transition that happened."""
        if pd is not None and pd.allow:              # would_deny / policy_error: never committed
            try:
                self.policy.commit(pd)
                fields["committed"] = True
            except Exception as e:                   # noqa: BLE001
                fields["commit_error"] = f"{type(e).__name__}: {e}"
                warn(f"policy commit failed for {pd.tool!r}: {e}")
        fields["states_after"] = self.policy.states()

    # ---------------- observation ----------------
    def register(self, msg, tool, args, extra=None):
        """Add a tools/call to pending BEFORE its bytes are sent. Returns the entry."""
        key = id_key(msg["id"])
        entry = _Pending(key, self.seq, time.monotonic_ns(), tool, args, extra)
        queue = self.pending.setdefault(key, [])
        if queue:                                    # same id already in flight
            warn(f"duplicate in-flight request id {key}: responses matched in order, records marked")
            entry.duplicate = True
            for other in queue:
                other.duplicate = True
        queue.append(entry)
        self.seq += 1
        return entry

    def remove(self, entry):
        queue = self.pending.get(entry.key)
        if queue and entry in queue:
            queue.remove(entry)
            if not queue:
                del self.pending[entry.key]

    def mark_dispatched(self, entry):
        entry.dispatched = True
        if entry.response is not None:               # response already arrived: log now
            self.finish(entry)

    def finish(self, entry):
        """Write the record, THEN drop the entry (a failed write leaves it pending)."""
        self.write_record(entry)
        self.remove(entry)

    def write_record(self, entry):
        t_resp, result, text, is_error = entry.response
        extra = entry.extra
        if entry.duplicate:
            extra = {**(extra or {}), "duplicate_id": True}
        # `result` is required: events.py hashes it into res_h (guide §8.4).
        self.log.write(seq=entry.seq, t_req=entry.t_req, t_resp=t_resp, tool=entry.tool,
                       args=entry.args, is_error=is_error, result_text=text, result=result,
                       extra=extra)

    @staticmethod
    def is_call(msg):
        return msg is not None and msg.get("method") == "tools/call" and "id" in msg

    def observe_down(self, line):
        msg = parse_copy(line)
        if msg is None or "id" not in msg or ("result" not in msg and "error" not in msg):
            return
        key = id_key(msg["id"])
        if self.policy is not None and key in self.list_ids:
            self.check_tool_names(msg)
            return
        queue = self.pending.get(key)
        if not queue:                    # response to some other method (initialize, ...)
            return
        entry = next((e for e in queue if e.response is None), None)
        if entry is None:
            warn(f"extra response for request id {key} with no call waiting; not logged")
            return
        result, text, is_error = result_text_and_error(msg)
        entry.response = (time.monotonic_ns(), result, text, is_error)
        if entry.dispatched:
            self.finish(entry)
        # else: held until the up pump finishes dispatch (and commit)

    def check_tool_names(self, msg):
        """Warn about policy tool names the server does not offer (typos never match)."""
        self.list_ids.discard(id_key(msg["id"]))
        result = msg.get("result")
        if not isinstance(result, dict):
            return
        self.listed_tools |= {t.get("name") for t in result.get("tools", []) if isinstance(t, dict)}
        if result.get("nextCursor"):     # more pages to come
            return
        missing = sorted(self.policy.named_tools() - self.listed_tools)
        if missing:
            warn(f"policy names tools the server does not offer: {missing}")

    # ---------------- pumps ----------------
    async def up_pump(self, stdin_reader, child):
        """Our stdin -> child's stdin."""
        try:
            while True:
                line = await read_line(stdin_reader)
                if not line:                         # §8.5 step 1: our stdin hit EOF
                    break
                self.tee.write("up_in", line)
                msg = parse_copy(line)
                is_call = self.is_call(msg)
                pd, fields, entry = None, None, None
                if is_call:
                    tool, args, problem = split_call(msg)    # never raises on malformed input
                    if problem is not None:
                        warn(f"{problem}; forwarded unchanged")
                if self.policy is not None:
                    if is_call:
                        pd, fields = self.check(tool, problem)
                    elif msg is not None and msg.get("method") == "tools/list" and "id" in msg:
                        self.list_ids.add(id_key(msg["id"]))
                    elif msg is not None and msg.get("method") == "tools/call":
                        warn("tools/call without an id forwarded WITHOUT a policy decision")
                if is_call:
                    entry = self.register(msg, tool, args, fields)  # BEFORE sending
                try:
                    child.stdin.write(line)          # ORIGINAL bytes, never re-serialised
                    await child.stdin.drain()        # raises if the bytes could not be sent
                except BaseException:
                    if entry is not None:
                        self.remove(entry)           # never dispatched: no commit, no record
                    raise
                self.tee.write("up_out", line)
                if is_call:
                    if self.policy is not None:
                        self.commit(pd, fields)      # only after successful dispatch
                    self.mark_dispatched(entry)      # writes the record if the response came first
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


async def run(args, policy):
    log = ChainWriter(args.log)
    tee = Tee(args.tee)
    proxy = Proxy(log, tee, policy, args.mode)

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
        if proxy.pending_count():
            warn(f"{proxy.pending_count()} tools/call request(s) not logged (no response, or log write failed)")
        log.close()                                  # §8.5 step 4
        tee.close()
    return code


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    policy = None
    if args.policy is not None:
        # Validate BEFORE creating the log or starting the server.
        enabled = None if args.rules is None else [r.strip() for r in args.rules.split(",") if r.strip()]
        try:
            policy = load_policy(args.policy, enabled)
        except PolicyError as e:
            warn(f"invalid policy, server NOT started: {e}")
            sys.exit(EXIT_BAD_POLICY)
        warn(f"policy {args.policy} ({policy.fingerprint[:12]}) rules={list(policy.enabled)} mode={args.mode}")
    code = asyncio.run(run(args, policy))
    sys.exit(code)                                   # §8.5 step 5: child's exit code


if __name__ == "__main__":
    main()
