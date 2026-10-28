"""mcpfw policy rules. Standard library only.

Step 1: R1 "untrusted read -> egress", as an explicit deterministic finite
automaton (DFA). Not yet connected to the proxy.

How R1 works
------------
Every tool call is put in one class:
    SOURCE       reads untrusted (attacker-controllable) content
    SINK         sends money/data out (egress)
    SOURCE_SINK  both (e.g. Slack get_webpage: returns untrusted text, and
                 requesting a URL could carry data out)
    OTHER        anything else, including tools the rule has never heard of

Two states:
    CLEAN    no untrusted content read yet in this session
    TAINTED  untrusted content has been read

Transition table (state, class) -> (decision, next state):

                 SOURCE           SINK             SOURCE_SINK      OTHER
    CLEAN        allow, TAINTED   allow, CLEAN     allow, TAINTED   allow, CLEAN
    TAINTED      allow, TAINTED   DENY,  TAINTED   DENY,  TAINTED   allow, TAINTED

Decide / commit lifecycle
-------------------------
    d = rule.decide(tool, server)   # pure: never changes state
    if d.allow:
        <proxy forwards the call>
        rule.commit(d)               # apply exactly the transition that was authorised
commit() takes the Decision itself, not the tool name, so it applies the
transition that was authorised and nothing else. It refuses a Decision that
is denied, issued by a different rule object, or STALE (the rule's state
changed after the decision was made).

Proxy responsibility (cannot be enforced here): the policy cannot see the
wire, so it cannot prove a call was really forwarded. The proxy must call
commit() only after it has actually dispatched the request, and never for a
call it dropped. This is tested at the integration level (planned step 4).

Limits of R1, to state in the report
------------------------------------
* Coarse, tool-level policy: tools are classified by NAME only. R1 does not
  look at arguments, URL contents, or whether sensitive data is present, so
  it does not detect actual data leakage; it blocks egress-class calls after
  any untrusted read. A denied get_webpage, for example, is blocked because
  of its class, not because its URL was found to contain stolen data.
* Tool mapping is experiment configuration, passed in, never hard-coded:
    - SOURCES are measured: a canary probe injects marker text into every
      AgentDojo injection point and records which tools return it.
    - SINKS are a policy choice (tools that move money/data out), documented
      with the experiment config and evaluated through the false-positive rate.
* `cross_server` is DIAGNOSTIC METADATA ONLY. It records whether a sink call
  goes to a different server than the untrusted reads, never changes a
  decision, and is not cross-server exfiltration detection.
"""
import hashlib
import json
from dataclasses import dataclass, field

CLEAN = "CLEAN"
TAINTED = "TAINTED"

SOURCE = "SOURCE"
SINK = "SINK"
SOURCE_SINK = "SOURCE_SINK"
OTHER = "OTHER"

ALLOW = True
DENY = False

# The DFA. Every (state, class) pair is listed, so nothing is implicit.
TRANSITIONS = {
    (CLEAN, SOURCE): (ALLOW, TAINTED),
    (CLEAN, SINK): (ALLOW, CLEAN),
    (CLEAN, SOURCE_SINK): (ALLOW, TAINTED),
    (CLEAN, OTHER): (ALLOW, CLEAN),
    (TAINTED, SOURCE): (ALLOW, TAINTED),
    (TAINTED, SINK): (DENY, TAINTED),
    (TAINTED, SOURCE_SINK): (DENY, TAINTED),
    (TAINTED, OTHER): (ALLOW, TAINTED),
}


class StaleDecisionError(ValueError):
    """The rule's state changed after this decision was made."""


@dataclass(frozen=True)
class Decision:
    allow: bool
    rule: str                 # rule id, e.g. "R1"
    tool: str
    server: str | None
    tool_class: str           # SOURCE / SINK / SOURCE_SINK / OTHER
    state_before: str
    state_after: str          # state if this call is committed (unchanged on deny)
    reason: str               # deny text shown to the agent; "" when allowed
    cross_server: bool | None = None
    """Diagnostic only. For a SINK/SOURCE_SINK call after a taint: True if it
    goes to a server no untrusted read came from, False if same server, None
    if unknown or not applicable."""
    destination: str | None = None
    """R2 only: the normalised destination this call sends to (None if the
    call has no destination or it was missing/malformed). R1 never sets it."""
    version: int = 0         # rule's commit count when decided (staleness check)
    _issuer: object = field(default=None, repr=False, compare=False)


@dataclass
class _Taint:
    tool: str
    server: str | None


@dataclass
class TaintThenDeny:
    """Generic rule: after any SOURCE call, deny every SINK call.

    R1 = TaintThenDeny("R1", sources=<untrusted reads>, sinks=<egress tools>).
    The same class can later serve R3 with a different sink set.
    """

    rule_id: str
    sources: frozenset
    sinks: frozenset
    state: str = field(default=CLEAN, init=False)
    taints: list = field(default_factory=list, init=False)   # every committed SOURCE call
    version: int = field(default=0, init=False)              # +1 per commit

    def __post_init__(self):
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("rule_id must be a non-empty string")
        self.sources = self._tool_set(self.sources, "sources")
        self.sinks = self._tool_set(self.sinks, "sinks")
        if not self.sources:
            raise ValueError(f"{self.rule_id}: sources is empty, so the rule could never trigger")
        if not self.sinks:
            raise ValueError(f"{self.rule_id}: sinks is empty, so the rule could never deny")

    @staticmethod
    def _tool_set(tools, name):
        if isinstance(tools, str):
            raise TypeError(f"{name} must be a collection of tool names, not a single string")
        tools = frozenset(tools)
        bad = [t for t in tools if not isinstance(t, str) or not t]
        if bad:
            raise TypeError(f"{name} contains invalid tool names: {bad!r}")
        return tools

    def classify(self, tool):
        is_src, is_sink = tool in self.sources, tool in self.sinks
        if is_src and is_sink:
            return SOURCE_SINK
        if is_sink:
            return SINK
        if is_src:
            return SOURCE
        return OTHER

    def decide(self, tool, server=None):
        """What would happen to this call. Does NOT change state."""
        cls = self.classify(tool)
        allow, next_state = TRANSITIONS[(self.state, cls)]

        cross = None
        if cls in (SINK, SOURCE_SINK) and self.taints and server is not None:
            taint_servers = {t.server for t in self.taints}
            if None not in taint_servers:
                cross = server not in taint_servers

        reason = "" if allow else self._reason(tool)
        return Decision(allow=allow, rule=self.rule_id, tool=tool, server=server, tool_class=cls,
                        state_before=self.state, state_after=next_state, reason=reason,
                        cross_server=cross, version=self.version, _issuer=self)

    def commit(self, decision):
        """Apply the transition authorised by `decision`.

        Call only after the proxy has actually forwarded the call.
        Raises ValueError for a denied decision or one from another rule,
        StaleDecisionError if the state changed since the decision was made.
        """
        if not isinstance(decision, Decision):
            raise TypeError("commit() takes the Decision returned by decide()")
        if decision._issuer is not self:
            raise ValueError(f"{self.rule_id}: decision was issued by a different rule")
        if not decision.allow:
            raise ValueError(f"{self.rule_id}: cannot commit denied call `{decision.tool}`")
        if decision.version != self.version or decision.state_before != self.state:
            raise StaleDecisionError(
                f"{self.rule_id}: decision for `{decision.tool}` is stale "
                f"(decided at version {decision.version}, rule is now at version {self.version})")
        if decision.tool_class in (SOURCE, SOURCE_SINK):
            self.taints.append(_Taint(decision.tool, decision.server))
        self.state = decision.state_after
        self.version += 1

    def _reason(self, tool):
        """Deterministic deny text: same calls in the same order -> same text."""
        first = self.taints[0]
        where = f" on server `{first.server}`" if first.server else ""
        return (f"Blocked by mcpfw ({self.rule_id}): `{tool}` is an egress-class tool, "
                f"and untrusted content was read earlier in this session "
                f"(first by `{first.tool}`{where}).")

    def snapshot(self):
        """State for logging."""
        return {"state": self.state, "version": self.version,
                "taints": [{"tool": t.tool, "server": t.server} for t in self.taints]}


# =====================================================================
# R2: at most K distinct destinations per session
# =====================================================================
def count_state(n):
    """State name for an R2 automaton that has used n distinct destinations."""
    return f"COUNT_{n}"


def normalise_destination(value):
    """'Alice', ' alice ' and 'ALICE' are the same destination."""
    return value.strip().casefold()


@dataclass
class DistinctDestinationLimit:
    """R2: a session may send to at most K distinct destinations.

    A destination is the value of one argument of an egress tool, e.g.
    send_money.recipient, send_channel_message.channel, post_webpage.url.

    States COUNT_0 .. COUNT_K (finite, because K is fixed). For a call:
        tool not in dest_args            -> allow (OTHER), state unchanged
        destination already used         -> allow, state unchanged
        new destination, count <  K      -> allow, COUNT_n -> COUNT_n+1
        new destination, count == K      -> DENY,  state unchanged
        destination argument missing     -> allow, not counted, if the tool is
                                            in `optional` (it may omit it, e.g.
                                            update_scheduled_transaction);
                                            otherwise DENY
        destination malformed (not a     -> DENY (fail closed)
        string, or blank), or args not a dict
    Example, K=2: A, A, B, C -> allow, allow, allow, DENY.

    Same decide/commit lifecycle and staleness check as R1: only forwarded
    (committed) calls use a slot.

    Limits, to state in the report:
    * Destinations are compared as normalised strings only. "www.x.com" and
      "https://www.x.com/" count as two destinations; two different strings
      that mean the same account are not merged.
    * R2 is not taint-gated: it counts destinations from the start of the
      session, whether or not untrusted content was read.
    """

    rule_id: str
    dest_args: dict           # tool name -> name of the argument holding the destination
    k: int
    optional: frozenset = frozenset()   # tools that may omit the destination argument
    destinations: list = field(default_factory=list, init=False)   # committed, normalised, in order
    version: int = field(default=0, init=False)

    def __post_init__(self):
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("rule_id must be a non-empty string")
        if not isinstance(self.dest_args, dict) or not self.dest_args:
            raise ValueError(f"{self.rule_id}: dest_args must be a non-empty dict tool -> argument name")
        bad = [(t, a) for t, a in self.dest_args.items()
               if not isinstance(t, str) or not t or not isinstance(a, str) or not a]
        if bad:
            raise TypeError(f"{self.rule_id}: dest_args has invalid entries: {bad!r}")
        self.dest_args = dict(self.dest_args)
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"{self.rule_id}: k must be an integer >= 1, got {self.k!r}")
        if isinstance(self.optional, str):
            raise TypeError("optional must be a collection of tool names, not a single string")
        self.optional = frozenset(self.optional)
        extra = self.optional - set(self.dest_args)
        if extra:
            raise ValueError(f"{self.rule_id}: optional names tools not in dest_args: {sorted(extra)}")

    @property
    def state(self):
        return count_state(len(self.destinations))

    def classify(self, tool):
        return SINK if tool in self.dest_args else OTHER

    def decide(self, tool, args, server=None):
        """What would happen to this call. Does NOT change state."""
        cls = self.classify(tool)
        before = self.state

        def result(allow, dest=None, after=before, reason=""):
            return Decision(allow=allow, rule=self.rule_id, tool=tool, server=server, tool_class=cls,
                            state_before=before, state_after=after, reason=reason,
                            destination=dest, version=self.version, _issuer=self)

        if cls == OTHER:
            return result(True)
        if not isinstance(args, dict):
            return result(False, reason=self._malformed(tool, "the call has no valid arguments"))
        arg = self.dest_args[tool]
        if arg not in args:
            if tool in self.optional:
                return result(True)
            return result(False, reason=self._malformed(tool, f"argument `{arg}` is missing"))
        raw = args[arg]
        if not isinstance(raw, str) or not raw.strip():
            return result(False, reason=self._malformed(tool, f"argument `{arg}` is not a usable destination"))

        dest = normalise_destination(raw)
        if dest in self.destinations:
            return result(True, dest)
        if len(self.destinations) < self.k:
            return result(True, dest, after=count_state(len(self.destinations) + 1))
        return result(False, dest, reason=self._limit_reason(tool, dest))

    def commit(self, decision):
        """Apply the transition authorised by `decision` (after forwarding)."""
        if not isinstance(decision, Decision):
            raise TypeError("commit() takes the Decision returned by decide()")
        if decision._issuer is not self:
            raise ValueError(f"{self.rule_id}: decision was issued by a different rule")
        if not decision.allow:
            raise ValueError(f"{self.rule_id}: cannot commit denied call `{decision.tool}`")
        if decision.version != self.version or decision.state_before != self.state:
            raise StaleDecisionError(
                f"{self.rule_id}: decision for `{decision.tool}` is stale "
                f"(decided at version {decision.version}, rule is now at version {self.version})")
        if decision.destination is not None and decision.destination not in self.destinations:
            self.destinations.append(decision.destination)
        self.version += 1

    def _limit_reason(self, tool, dest):
        used = ", ".join(f"`{d}`" for d in self.destinations)
        return (f"Blocked by mcpfw ({self.rule_id}): `{tool}` would send to a new destination "
                f"`{dest}`, but this session has already used the maximum of {self.k} "
                f"distinct destination(s) ({used}).")

    def _malformed(self, tool, what):
        return f"Blocked by mcpfw ({self.rule_id}): `{tool}` cannot be checked: {what}."

    def snapshot(self):
        """State for logging."""
        return {"state": self.state, "version": self.version, "k": self.k,
                "destinations": list(self.destinations)}


# =====================================================================
# Step 2: policy files and the multi-rule Policy
# =====================================================================
"""
Policy file format (JSON), version 1:

    {
      "version": 1,
      "suite": "banking",
      "classes": {"<class name>": ["tool", ...], ...},
      "rules":   {"<rule id>": {"type": "taint_then_deny",
                                "source": "<class name>", "sink": "<class name>"}, ...},
      "notes": "optional free text"
    }

Rules are evaluated in the order they appear in the file. Every mistake in a
file raises PolicyError: a broken policy must never silently mean "no
protection".
"""

POLICY_VERSION = 1
_TOP_KEYS_REQUIRED = {"version", "suite", "classes", "rules"}
_TOP_KEYS_OPTIONAL = {"notes"}
_RULE_TYPES = {
    # type -> (required keys besides "type", builder)
    "taint_then_deny": ({"source", "sink"},
                        lambda rid, spec, classes: TaintThenDeny(rid, classes[spec["source"]],
                                                                 classes[spec["sink"]])),
}


class PolicyError(ValueError):
    """The policy file or the enabled-rule selection is invalid."""


def _canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _reject_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise PolicyError(f"duplicate key {k!r} (JSON would silently keep only the last one)")
        out[k] = v
    return out


def _check_keys(obj, required, optional, where):
    if not isinstance(obj, dict):
        raise PolicyError(f"{where}: must be a JSON object")
    missing = required - obj.keys()
    unknown = obj.keys() - required - optional
    if missing:
        raise PolicyError(f"{where}: missing {sorted(missing)}")
    if unknown:
        raise PolicyError(f"{where}: unknown keys {sorted(unknown)} (typo?)")


def _validate(data):
    """Check the whole structure. Returns classes as {name: frozenset(tools)}."""
    _check_keys(data, _TOP_KEYS_REQUIRED, _TOP_KEYS_OPTIONAL, "policy")
    if data["version"] != POLICY_VERSION or isinstance(data["version"], bool):
        raise PolicyError(f"policy: unsupported version {data['version']!r} (expected {POLICY_VERSION})")
    if not isinstance(data["suite"], str) or not data["suite"]:
        raise PolicyError("policy: suite must be a non-empty string")
    if "notes" in data and not isinstance(data["notes"], str):
        raise PolicyError("policy: notes must be a string")

    if not isinstance(data["classes"], dict) or not data["classes"]:
        raise PolicyError("classes: must be a non-empty object")
    classes = {}
    for name, tools in data["classes"].items():
        if not name:
            raise PolicyError("classes: empty class name")
        if not isinstance(tools, list) or not tools:
            raise PolicyError(f"classes.{name}: must be a non-empty list of tool names")
        bad = [t for t in tools if not isinstance(t, str) or not t]
        if bad:
            raise PolicyError(f"classes.{name}: invalid tool names {bad!r}")
        dupes = sorted({t for t in tools if tools.count(t) > 1})
        if dupes:
            raise PolicyError(f"classes.{name}: duplicate tools {dupes}")
        classes[name] = frozenset(tools)

    if not isinstance(data["rules"], dict) or not data["rules"]:
        raise PolicyError("rules: must be a non-empty object")
    for rid, spec in data["rules"].items():
        if not rid:
            raise PolicyError("rules: empty rule id")
        if not isinstance(spec, dict) or "type" not in spec:
            raise PolicyError(f"rules.{rid}: must be an object with a 'type'")
        if spec["type"] not in _RULE_TYPES:
            raise PolicyError(f"rules.{rid}: unknown type {spec['type']!r}; known: {sorted(_RULE_TYPES)}")
        required, _ = _RULE_TYPES[spec["type"]]
        _check_keys(spec, required | {"type"}, set(), f"rules.{rid}")
        for key in sorted(required):
            if spec[key] not in classes:
                raise PolicyError(f"rules.{rid}.{key}: class {spec[key]!r} is not defined in classes")
    return classes


@dataclass(frozen=True)
class PolicyDecision:
    allow: bool
    tool: str
    server: str | None
    rules_fired: tuple        # ids of the rules that denied, in file order
    reason: str               # deny text for the agent ("" when allowed)
    decisions: tuple          # per-rule Decision objects, in file order
    _issuer: object = field(default=None, repr=False, compare=False)

    def states_before(self):
        return {d.rule: d.state_before for d in self.decisions}

    def states_after(self):
        """States after commit if allowed; unchanged if denied."""
        if not self.allow:
            return self.states_before()
        return {d.rule: d.state_after for d in self.decisions}


class Policy:
    """The enabled rules of one policy file, evaluated together.

    decide(): asks EVERY enabled rule; denied if ANY rule denies.
    commit(): all-or-nothing. Checks that every rule's decision is still
              current BEFORE changing any rule, so rules never get out of sync.
    """

    def __init__(self, rules, *, suite, fingerprint, enabled, source):
        self.rules = rules                  # {rule_id: rule}, file order
        self.suite = suite
        self.fingerprint = fingerprint      # sha256 of the file's canonical JSON
        self.enabled = enabled              # tuple of enabled rule ids, file order
        self.source = source

    def decide(self, tool, server=None):
        decisions = tuple(rule.decide(tool, server) for rule in self.rules.values())
        denied = [d for d in decisions if not d.allow]
        return PolicyDecision(
            allow=not denied, tool=tool, server=server,
            rules_fired=tuple(d.rule for d in denied),
            reason=" ".join(d.reason for d in denied),
            decisions=decisions, _issuer=self)

    def commit(self, pdecision):
        if not isinstance(pdecision, PolicyDecision):
            raise TypeError("commit() takes the PolicyDecision returned by decide()")
        if pdecision._issuer is not self:
            raise ValueError("decision was issued by a different policy")
        if not pdecision.allow:
            raise ValueError(f"cannot commit denied call `{pdecision.tool}`")
        # Check everything first ...
        for d in pdecision.decisions:
            rule = self.rules[d.rule]
            if d._issuer is not rule:
                raise ValueError(f"{d.rule}: decision was issued by a different rule")
            if d.version != rule.version or d.state_before != rule.state:
                raise StaleDecisionError(f"{d.rule}: decision for `{d.tool}` is stale")
        # ... then change everything.
        for d in pdecision.decisions:
            self.rules[d.rule].commit(d)

    def snapshot(self):
        return {rid: rule.snapshot() for rid, rule in self.rules.items()}

    def states(self):
        return {rid: rule.state for rid, rule in self.rules.items()}

    def named_tools(self):
        """Every tool name the enabled rules refer to (for checking against tools/list)."""
        names = set()
        for rule in self.rules.values():
            names |= rule.sources | rule.sinks
        return names


def policy_from_dict(data, enabled=None, source="<dict>"):
    """Build a Policy from already-parsed JSON. enabled=None means all rules."""
    classes = _validate(data)
    all_ids = list(data["rules"])
    if enabled is None:
        chosen = all_ids
    else:
        if isinstance(enabled, str):
            raise PolicyError("enabled must be a list of rule ids, not a single string")
        enabled = list(enabled)
        if not enabled:
            raise PolicyError("no rules enabled (to run without enforcement, run mcpfw without a policy)")
        unknown = [r for r in enabled if r not in data["rules"]]
        if unknown:
            raise PolicyError(f"enabled rules not in the policy: {unknown}; available: {all_ids}")
        chosen = [r for r in all_ids if r in enabled]        # file order, deterministic
    rules = {}
    for rid in chosen:
        spec = data["rules"][rid]
        _, build = _RULE_TYPES[spec["type"]]
        rules[rid] = build(rid, spec, classes)
    return Policy(rules, suite=data["suite"], fingerprint=hashlib.sha256(_canon(data)).hexdigest(),
                  enabled=tuple(chosen), source=source)


def load_policy(path, enabled=None):
    """Read and validate a policy file. Raises PolicyError on any problem."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_reject_duplicate_keys)
    except OSError as e:
        raise PolicyError(f"{path}: cannot read ({e})") from None
    except json.JSONDecodeError as e:
        raise PolicyError(f"{path}: invalid JSON ({e})") from None
    except PolicyError as e:
        raise PolicyError(f"{path}: {e}") from None
    return policy_from_dict(data, enabled, source=str(path))
