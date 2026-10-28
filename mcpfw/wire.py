"""Wire helpers for mcpfw: line reading and --tee capture (guide §8.2, §8.6).

Standard library only.
"""
import asyncio
import os

LINE_LIMIT = 16 * 1024 * 1024      # 16 MiB (asyncio's default 64 KiB is too small)


async def read_line(reader):
    """Read one newline-terminated line, exactly as sent.

    Returns b"" at EOF. If the stream ends without a final newline, returns
    the remaining bytes unchanged. Lines longer than LINE_LIMIT are still read
    whole (in chunks), so nothing is ever cut or altered.
    """
    parts = []
    while True:
        try:
            parts.append(await reader.readuntil(b"\n"))
            return b"".join(parts)
        except asyncio.IncompleteReadError as e:      # EOF
            parts.append(e.partial)
            return b"".join(parts)
        except asyncio.LimitOverrunError as e:        # longer than the limit
            parts.append(await reader.readexactly(e.consumed))


class Tee:
    """Appends raw bytes to capture files (§8.6). Disabled when dir is None.

    up_in        every line read from the agent
    up_out       lines forwarded to the server
    up_blocked   lines NOT forwarded because enforce mode blocked them
    down_in      every line read from the server
    down_out     every line written to the agent
    down_injected  block responses mcpfw itself wrote to the agent
    Without blocking, up_blocked and down_injected stay empty.
    """

    NAMES = ("up_in", "up_out", "up_blocked", "down_in", "down_out", "down_injected")

    def __init__(self, directory):
        self._files = {}
        if directory:
            os.makedirs(directory, exist_ok=True)
            for n in self.NAMES:
                self._files[n] = open(os.path.join(directory, f"{n}.bin"), "ab")

    def write(self, name, data):
        f = self._files.get(name)
        if f is not None:
            f.write(data)
            f.flush()

    def close(self):
        for f in self._files.values():
            f.close()


class WireMismatch(Exception):
    pass


def _lines(path):
    return path.read_bytes().splitlines(keepends=True) if path.exists() else []


def _merge_check(total, a, b, what):
    """Every line of `total`, in order, must be the next line of `a` or of `b`,
    and every line of `a` and `b` must be used exactly once."""
    i = j = 0
    for n, line in enumerate(total):
        if i < len(a) and line == a[i]:
            i += 1
        elif j < len(b) and line == b[j]:
            j += 1
        else:
            raise WireMismatch(f"{what}: line {n + 1} is not accounted for: {line[:80]!r}")
    if i != len(a) or j != len(b):
        raise WireMismatch(f"{what}: {len(a) - i + len(b) - j} captured line(s) never appeared")


def verify_tee(directory):
    """X2 under enforce: every byte accounted for, in order.

        up_in    = up_out   interleaved with up_blocked
        down_out = down_in  interleaved with down_injected
    With no blocking this is exactly the strict X2 equality.
    """
    from pathlib import Path
    d = Path(directory)
    _merge_check(_lines(d / "up_in.bin"), _lines(d / "up_out.bin"), _lines(d / "up_blocked.bin"),
                 "agent->server")
    _merge_check(_lines(d / "down_out.bin"), _lines(d / "down_in.bin"), _lines(d / "down_injected.bin"),
                 "server->agent")
