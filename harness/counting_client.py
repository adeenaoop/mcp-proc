"""Counting wrappers for direct-mode runs (guide §3.3), AgentDojo 0.1.35.

Two pieces, neither of which modifies AgentDojo:

CountingClient   Wraps an OpenAI client. AgentDojo's OpenAILLM calls
                 client.chat.completions.create(...); this wrapper forwards
                 that call unchanged and adds the response's token usage to
                 counters. Everything else on the client passes straight through.

CountingRuntime  A FunctionsRuntime subclass that counts tool executions.
                 Needed because run_task_with_pipeline() returns only
                 (utility, security) and discards the messages, so the tool
                 call count cannot be read from them as §3.4 suggests.
                 run_task_with_pipeline() accepts runtime_class=, so we pass
                 this class in and it counts every run_function() call.
"""
from agentdojo.functions_runtime import FunctionsRuntime


class _Completions:
    def __init__(self, real, counter):
        self._real = real
        self._counter = counter

    def create(self, *args, **kwargs):
        response = self._real.create(*args, **kwargs)   # forwarded unchanged
        usage = getattr(response, "usage", None)
        self._counter.llm_calls += 1
        if usage is not None:
            self._counter.tok_in += usage.prompt_tokens or 0
            self._counter.tok_out += usage.completion_tokens or 0
        return response

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Chat:
    def __init__(self, real, counter):
        self._real = real
        self.completions = _Completions(real.completions, counter)

    def __getattr__(self, name):
        return getattr(self._real, name)


class CountingClient:
    """Drop-in stand-in for openai.OpenAI that counts calls and tokens."""

    def __init__(self, client):
        self._client = client
        self.reset()
        self.chat = _Chat(client.chat, self)

    def reset(self):
        """Call at the start of every episode."""
        self.llm_calls = 0
        self.tok_in = 0
        self.tok_out = 0

    def __getattr__(self, name):
        return getattr(self._client, name)


class CountingRuntime(FunctionsRuntime):
    """FunctionsRuntime that counts tool executions.

    AgentDojo creates the runtime itself (runtime_class(suite.tools)), so we
    can't hold a reference to the instance; the count lives on the class and
    is reset at the start of each episode.
    """

    n_tool_calls = 0

    @classmethod
    def reset(cls):
        cls.n_tool_calls = 0

    def run_function(self, env, function, kwargs, raise_on_error=False):
        type(self).n_tool_calls += 1
        return super().run_function(env, function, kwargs, raise_on_error)
