"""mcpfw: MCP stdio proxy with a hash-chained call log (guide §8).

Three ways to run it:

  no --policy      Pass-through. Every byte is forwarded unchanged in both
                   directions (X2 proves this) and each tools/call is logged.
                   The log format is the Week 1 format, with one addition: two
                   in-flight calls with the same id are both marked
                   "duplicate_id": true. (The internal bookkeeping changed in
                   step 3 to fix a logging race and a crash on malformed
                   params; the bytes on the wire did not change.)

  --mode audit     (default with --policy) The policy decides each tools/call
                   BEFORE it is forwarded; the decision is logged. Nothing is
                   blocked: a call the policy would deny is forwarded, logged
                   as "would_deny", and NOT committed, so the log shows what
                   enforcement would decide on this exact trace.

  --mode enforce   Same decisions, but a denied call is NOT forwarded. mcpfw
                   answers the agent itself with a tool error carrying the
                   block reason (same JSON-RPC id), so the agent can continue.
                   Fail closed: a call the policy cannot decide (malformed
                   call, policy failure) is blocked too.

Log fields with a policy: "decision" is what the policy said (allow,
would_deny, deny, policy_error); "action" is what mcpfw did (forward, block).

Lifecycle per tools/call:
    parse a copy -> [decide] -> register as pending -> forward ORIGINAL bytes
    -> [commit] -> mark dispatched
The call is registered BEFORE its bytes are sent, so a fast server's response
can never arrive for a call the proxy does not know about. Its log record is
written only when BOTH the response has arrived AND dispatch has finished
(commit outcome known); a response that comes first is held until then.

Commit criterion = successful DISPATCH: the request bytes were written and
drain() returned. This does NOT mean the tool ran successfully; drain() only
says the bytes left the proxy's buffer. If writing fails, the call is not
confirmed as dispatched: it is dropped from pending and never committed
(this does not prove the server received nothing).

A blocked call is never registered as pending (no server response will come);
its record is written immediately, BEFORE the block response is sent, so no
block ever happens without a log record.

Not covered yet (step 5): JSON arrays (batches) and tools/call messages
without an id are forwarded WITHOUT a policy decision (a warning is printed),
in every mode; duplicate in-flight ids are decided and forwarded normally but
not rejected (records marked "duplicate_id"). Coverage is therefore
"tools/call requests with an id", not every possible message.

Malformed calls never crash the proxy: params not an object, arguments not an
object (null counts as "no arguments"), missing/invalid tool name, or a request
id that is not a string or integer (MCP's rule). Audit: forwarded unchanged,
logged as "policy_error". Enforce: blocked, logged as "policy_error" (an
invalid id gets a JSON-RPC "Invalid Request" error with id null, because an
invalid id cannot be echoed). Never "allow".

Failure rules (fail closed):
  * Policy health. If the policy machinery fails (a commit raises, or its state
    cannot be read), the policy is marked FAILED for the rest of the session:
    its state can no longer be trusted. Enforce: every later tools/call is
    blocked as policy_error. Audit: logged as policy_error, forwarded. The call
    whose commit failed is still logged as forwarded (it was already sent).
  * Audit log. If a log record cannot be written, the session is stopped:
    nothing more is forwarded in either direction, the server's stdin is
    closed, and mcpfw exits with code 3. A blocked call is never forwarded and
    never answered without its record.
    Limit: a forwarded call's record is written when its RESPONSE arrives, so a
    failure is detected then. Calls a client already sent before that moment
    (pipelining) were forwarded and cannot be recalled. With a sequential client
    (one call, wait for the reply, next call -- as AgentDojo's agent does), no
    call is ever forwarded after a failed record.

Usage:
    python mcpfw/proxy.py --log run/<ep>/trace.jsonl [--tee DIR] \
        [--policy policies/banking.json [--rules R1,R2] [--mode audit|enforce]] \
        -- <server command and args...>

Exit codes: the server's exit code; 2 if the policy file or --rules is invalid
(checked BEFORE the server is started, so no server ever runs unprotected by
mistake); 3 if the audit log could not be written (session stopped).

Standard library only. RULE: never print to stdout (it carries the protocol).
"""
import argparse
import asyncio
import json
import signal
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
EXIT_LOG_FAILURE = 3
EXIT_TERMINATED = 143                                # 128 + SIGTERM, the shell convention


class SessionAbort(Exception):
    """The audit log could not be written: stop the protected session."""


def warn(*args):
    print("[mcpfw]", *args, file=sys.stderr, flush=True)


def parse_args(argv):
    if "--" not in argv:
        sys.exit("usage: mcpfw --log FILE [--tee DIR] [--policy FILE [--rules R1,..] [--mode audit|enforce]] -- <server...>")
    split = argv.index("--")
    p = argparse.ArgumentParser(prog="mcpfw")
    p.add_argument("--log", required=True, help="trace.jsonl path")
    p.add_argument("--tee", default=None, help="directory for raw byte captures (X2)")
    p.add_argument("--policy", default=None, help="policy JSON file; omit for pure pass-through")
    p.add_argument("--rules", default=None, help="comma-separated rule ids to enable (default: all in file)")
    p.add_argument("--mode", default=None, choices=["audit", "enforce"],
                   help="audit = decide and log, never block (default); enforce = block denied calls")
    p.add_argument("--stats", default=None,
                   help="at exit, write peak memory (proxy and server) to this JSON file")
    args = p.parse_args(argv[:split])
    if args.rules is not None and args.policy is None:
        p.error("--rules needs --policy")
    if args.mode is not None and args.policy is None:
        p.error("--mode needs --policy")
    if args.mode is None:
        args.mode = "audit"
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


def block_response(msg_id, reason):
    """The tool error mcpfw sends instead of forwarding a blocked call.
    Same shape as a server's tool error, so MCPRuntime/AgentDojo handle it
    normally; the id is echoed exactly as received (number or string)."""
    result = {"content": [{"type": "text", "text": reason}], "isError": True}
    line = json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    return result, line


def valid_request_id(msg_id):
    """MCP: a request id must be a string or an integer (not null, bool, float,
    array or object)."""
    return isinstance(msg_id, str) or (isinstance(msg_id, int) and not isinstance(msg_id, bool))


def invalid_request_response(reason):
    """JSON-RPC 'Invalid Request' error. id is null because an invalid id
    cannot be echoed back (JSON-RPC 2.0, section 5)."""
    error = {"code": -32600, "message": reason}
    line = json.dumps({"jsonrpc": "2.0", "id": None, "error": error},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    return error, line


NO_DECISION_REASON = ("Blocked by mcpfw: no policy decision could be made for this call ({error}). "
                      "In enforce mode, calls the policy cannot decide are blocked.")


def id_key(msg_id):
    return json.dumps(msg_id)            # keeps 1 and "1" distinct, exactly as parsed


def split_call(msg):
    """(tool, args, problem) from a tools/call message. Never raises.

    problem is None for a well-formed call, otherwise a short description
    (invalid id, malformed params or arguments, invalid tool name). Malformed
    calls get no real policy decision: audit forwards them, enforce blocks them.
    """
    if not valid_request_id(msg.get("id")):
        return None, None, f"invalid request id {msg.get('id')!r}: MCP requires a string or integer"
    params = msg.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None, None, f"malformed params: expected an object, got {type(params).__name__}"
    tool = params.get("name")
    args = params.get("arguments")
    if args is None:                     # omitted or null: "no arguments" (MCP allows omitting it)
        args = {}
    if not isinstance(tool, str) or not tool:
        return tool, args, f"invalid tool name {tool!r}"
    if not isinstance(args, dict):
        return tool, args, f"malformed arguments: expected an object, got {type(args).__name__}"
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
    def __init__(self, log, tee, policy=None, mode="audit", out=None):
        self.log = log
        self.tee = tee
        self.policy = policy
        self.mode = mode
        self.out = out                   # our stdout (needed to answer blocked calls)
        # request id key -> FIFO list of _Pending. A list, not a single entry,
        # so two in-flight calls with the same id never overwrite each other.
        self.pending = {}
        self.seq = 0
        self.list_ids = set()            # ids of tools/list requests (for the name check)
        self.listed_tools = set()
        self.policy_failure = None       # set once the policy machinery fails: state untrusted
        self.fatal = None                # set when the audit log fails: session stops
        self.up_task = None              # so a fatal error in the down pump can stop the up pump

    # ---------------- failure handling ----------------
    def mark_unhealthy(self, why):
        if self.policy_failure is None:
            self.policy_failure = why
            warn(f"POLICY FAILED, state no longer trusted: {why}. "
                 + ("Every later call will be BLOCKED." if self.mode == "enforce"
                    else "Later calls logged as policy_error."))

    def safe_states(self, fields):
        """Policy state for logging. Never raises: a failure is recorded in the
        log and marks the policy unhealthy."""
        try:
            return self.policy.states()
        except Exception as e:                       # noqa: BLE001
            fields["states_error"] = f"{type(e).__name__}: {e}"
            self.mark_unhealthy(f"cannot read policy state ({fields['states_error']})")
            return None

    def log_write(self, **kw):
        """Write one record. On failure: stop the session (fail closed)."""
        try:
            self.log.write(**kw)
        except Exception as e:                       # noqa: BLE001
            self.fatal = f"audit log write failed: {type(e).__name__}: {e}"
            warn(f"{self.fatal}. Stopping the session: nothing more is forwarded.")
            raise SessionAbort(self.fatal) from e

    def pending_count(self):
        return sum(len(q) for q in self.pending.values())

    # ---------------- policy (audit) ----------------
    def check(self, tool, problem=None):
        """Ask the policy about one call. Returns (PolicyDecision or None, log fields).
        Never raises: any failure becomes decision "policy_error".

        decision (what the policy said):
            allow         policy allows; committed after successful dispatch
            would_deny    (audit) policy would block; forwarded anyway, NOT committed
            deny          (enforce) policy blocks; NOT forwarded, NOT committed
            policy_error  no real decision was made (malformed call, policy failure,
                          or the policy is marked FAILED); NOT committed, never allow.
                          audit: forwarded; enforce: blocked (fail closed)
        action (what mcpfw does): "forward" or "block"
        """
        enforce = self.mode == "enforce"
        fields = {"mode": self.mode, "committed": False}
        states = None
        t0 = time.monotonic_ns()
        try:
            fields.update(policy_fp=self.policy.fingerprint, rules_enabled=list(self.policy.enabled))
            states = self.safe_states(fields)
            if problem is None and self.policy_failure is not None:
                problem = f"policy marked FAILED earlier in this session ({self.policy_failure})"
            if problem is not None:
                raise ValueError(problem)
            pd = self.policy.decide(tool)
            if pd.allow:
                decision, action = "allow", "forward"
            else:
                decision, action = ("deny", "block") if enforce else ("would_deny", "forward")
            fields.update(decision=decision, action=action,
                          rules_fired=list(pd.rules_fired), states_before=pd.states_before(),
                          states_after=pd.states_after(), reason=pd.reason)
        except Exception as e:                       # noqa: BLE001 (any failure -> policy_error)
            error = f"{type(e).__name__}: {e}"
            fields.update(decision="policy_error", action="block" if enforce else "forward",
                          rules_fired=[], states_before=states, states_after=states,
                          reason=NO_DECISION_REASON.format(error=error) if enforce else "",
                          policy_error=error)
            warn(f"no policy decision for tool {tool!r}: {error} "
                 f"({'blocked' if enforce else 'audit mode: forwarded'}, not committed)")
            pd = None
        fields["decide_ns"] = time.monotonic_ns() - t0
        fields["policy_health"] = "failed" if self.policy_failure else "ok"
        return pd, fields

    def block(self, msg, tool, args, fields):
        """Enforce: answer a blocked call ourselves. Log FIRST, then respond,
        so no block ever happens without a record."""
        fields["states_after"] = self.safe_states(fields)   # nothing committed
        if valid_request_id(msg.get("id")):
            result, line = block_response(msg["id"], fields["reason"])
        else:                                                  # cannot echo an invalid id
            result, line = invalid_request_response(fields["reason"])
        t = time.monotonic_ns()
        # Log FIRST. If this raises SessionAbort, no response is sent and the
        # request was never forwarded: nothing happens without a record.
        self.log_write(seq=self.seq, t_req=t, t_resp=t, tool=tool, args=args, is_error=True,
                       result_text=fields["reason"], result=result, extra=fields)
        self.seq += 1
        self.out.write(line)                                   # one whole line, then flush:
        self.out.flush()                                       # never interleaves with the down pump
        self.tee.write("down_out", line)
        self.tee.write("down_injected", line)

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
                # The call was already sent (still logged as forwarded), but the
                # rules did not record it: their state is now wrong. Stop trusting it.
                self.mark_unhealthy(f"commit failed for `{pd.tool}` ({fields['commit_error']})")
        fields["states_after"] = self.safe_states(fields)
        fields["policy_health"] = "failed" if self.policy_failure else "ok"

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
        self.log_write(seq=entry.seq, t_req=entry.t_req, t_resp=t_resp, tool=entry.tool,
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
            while self.fatal is None:
                line = await read_line(stdin_reader)
                if not line or self.fatal is not None:   # EOF, or session stopped meanwhile
                    break
                self.tee.write("up_in", line)
                msg = parse_copy(line)
                is_call = self.is_call(msg)
                pd, fields, entry = None, None, None
                if is_call:
                    tool, args, problem = split_call(msg)    # never raises on malformed input
                    if problem is not None and self.mode != "enforce":
                        warn(f"{problem}; forwarded unchanged")
                if self.policy is not None:
                    if is_call:
                        pd, fields = self.check(tool, problem)
                    elif msg is not None and msg.get("method") == "tools/list" and "id" in msg:
                        self.list_ids.add(id_key(msg["id"]))
                    elif msg is not None and msg.get("method") == "tools/call":
                        warn("tools/call without an id forwarded WITHOUT a policy decision")
                if is_call and fields is not None and fields["action"] == "block":
                    self.tee.write("up_blocked", line)   # NOT forwarded to the server
                    self.block(msg, tool, args, fields)
                    continue
                if is_call:
                    entry = self.register(msg, tool, args, fields)  # BEFORE sending
                try:
                    child.stdin.write(line)          # ORIGINAL bytes, never re-serialised
                    await child.stdin.drain()        # raises if the bytes could not be sent
                except BaseException:
                    if entry is not None:
                        self.remove(entry)           # not confirmed dispatched: no commit, no record
                    raise
                self.tee.write("up_out", line)
                if is_call:
                    if self.policy is not None:
                        self.commit(pd, fields)      # only after successful dispatch
                    self.mark_dispatched(entry)      # writes the record if the response came first
        except (BrokenPipeError, ConnectionResetError):
            warn("child stdin closed early")
        except SessionAbort:
            pass                                     # fatal: stop forwarding; finally closes the server
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
            if self.fatal is not None:               # session stopped: drain, forward nothing
                continue
            stdout.write(line)                       # exact bytes read
            stdout.flush()
            self.tee.write("down_out", line)
            try:
                self.observe_down(line)              # after forwarding
            except SessionAbort:
                if self.up_task is not None:
                    self.up_task.cancel()            # stop reading the agent; server stdin closes


async def run(args, policy):
    log = ChainWriter(args.log)
    tee = Tee(args.tee)
    proxy = Proxy(log, tee, policy, args.mode, out=sys.stdout.buffer)

    loop = asyncio.get_running_loop()
    # SIGTERM (e.g. the MCP client's shutdown escalation): stop in order, so the
    # finally block below still closes the log and the captures, instead of the
    # process dying mid-write. Records already written stay valid either way.
    main_task = asyncio.current_task()
    child = None
    try:
        loop.add_signal_handler(signal.SIGTERM, main_task.cancel)
    except (NotImplementedError, RuntimeError):     # not available on Windows
        pass
    stdin_reader = asyncio.StreamReader(limit=LINE_LIMIT)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stdin_reader), sys.stdin)

    child = await asyncio.create_subprocess_exec(
        *args.child, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=None, limit=LINE_LIMIT)              # stderr passes straight through

    stdout = sys.stdout.buffer
    up = asyncio.create_task(proxy.up_pump(stdin_reader, child))
    proxy.up_task = up
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
        if child is not None and child.returncode is None and main_task.cancelling():
            try:
                child.terminate()                    # we were told to stop: pass it on
            except ProcessLookupError:
                pass
        if proxy.pending_count():
            warn(f"{proxy.pending_count()} tools/call request(s) not logged (no response, or log write failed)")
        try:
            log.close()                              # §8.5 step 4
        except Exception as e:                       # noqa: BLE001 (log already failed)
            warn(f"closing the log failed: {e}")
        tee.close()
    if proxy.fatal is not None:
        warn(f"session stopped: {proxy.fatal}")
        return EXIT_LOG_FAILURE
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
    try:
        code = asyncio.run(run(args, policy))
    except asyncio.CancelledError:                   # only our SIGTERM handler cancels run()
        warn("stopped by SIGTERM; log and captures closed")
        code = EXIT_TERMINATED
    if args.stats:
        write_stats(args.stats, code)
    sys.exit(code)                                   # §8.5 step 5: child's exit code


def peak_rss_kb():
    """Peak resident memory of THIS process, in KB, and where it came from.

    Linux: VmHWM from /proc/self/status. It is reset when a program starts, so
    it is mcpfw's own peak. getrusage().ru_maxrss is NOT used on Linux because
    it also carries the parent's memory from the moment mcpfw was started (a
    large harness made mcpfw look like 200 MB instead of ~23 MB).
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]), "VmHWM"
    except OSError:
        pass
    import resource                                   # other systems: best available
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "ru_maxrss"


def write_stats(path, code):
    """Written after the session has ended (log and captures closed), so its
    existence also means the capture files are complete. Best effort: a
    failure only warns and never changes the exit code, because measurement
    must not affect the run.
    """
    try:
        import os
        kb, source = peak_rss_kb()
        data = {"proxy_peak_rss_kb": kb, "rss_source": source, "exit_code": code}
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)                        # atomic: readers never see half a file
    except Exception as e:                           # noqa: BLE001
        warn(f"could not write --stats file: {e}")


if __name__ == "__main__":
    main()
