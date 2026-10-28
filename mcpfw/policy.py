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
    version: int = 0          # rule's commit count when decided (staleness check)
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
