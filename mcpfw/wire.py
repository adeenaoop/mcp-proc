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
    """Appends raw bytes to four capture files (§8.6). Disabled when dir is None."""

    NAMES = ("up_in", "up_out", "down_in", "down_out")

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
