"""Fake OpenAI client that follows a task's ground truth. No key, no cost.

Used to test the episode runner end to end without a real model. On each
chat.completions.create() call it returns the next ground-truth tool call;
when none are left, it returns the task's GROUND_TRUTH_OUTPUT as a final
answer. A correct runner should therefore score utility=True on every task.

Token usage is fixed made-up numbers (100 in, 10 out per call); it only
proves the counters work, it is NOT a cost measurement.
"""
import json

from openai.types.chat import ChatCompletion


class _Completions:
    def __init__(self, owner):
        self._o = owner

    def create(self, **kwargs):
        o = self._o
        if o.i < len(o.calls):
            c = o.calls[o.i]
            o.i += 1
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{o.i}", "type": "function",
                "function": {"name": c.function, "arguments": json.dumps(dict(c.args))}}]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": o.final_text}
            finish = "stop"
        return ChatCompletion.model_validate({
            "id": "fake", "object": "chat.completion", "created": 0, "model": kwargs.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        })


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class GroundTruthFakeClient:
    """Plays back `calls` (list of FunctionCall), then answers `final_text`."""

    def __init__(self, calls, final_text):
        self.calls = list(calls)
        self.final_text = final_text
        self.i = 0
        self.chat = _Chat(self)
