"""Log records and hash chain for mcpfw (guide §8.4). Standard library only.

canon(x): JSON with sorted keys, no spaces, ensure_ascii=False, UTF-8 bytes.

Each record (Week 1 subset):
    seq, t_req, t_resp, tool, args, is_error, result_text,
    args_h = sha256(canon(args)), res_h = sha256(canon(result)), prev, h

Chain:
    prev of the first record = 64 zeros
    h = sha256(prev_bytes + canon(record without "h")), as hex
    prev of each next record = previous record's h

Choice recorded for the report: prev_bytes = bytes.fromhex(prev), i.e. the
32 raw hash bytes (the guide does not say raw bytes vs. hex text).

One canon line per record, flushed immediately, so every record written
before a crash stays valid.
"""
import hashlib
import json
import os

GENESIS = "0" * 64


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def chain_hash(prev, record_without_h):
    return sha256_hex(bytes.fromhex(prev) + canon(record_without_h))


class ChainWriter:
    """Appends hash-chained records to a JSONL file."""

    def __init__(self, path):
        # Refuse to overwrite an existing log: it is evidence, and a new chain
        # starting from zeros in the same file would break verification.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            raise FileExistsError(f"log already exists and is not empty: {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "wb")
        self._prev = GENESIS
        self.count = 0

    def write(self, *, seq, t_req, t_resp, tool, args, is_error, result_text, result):
        record = {
            "seq": seq,
            "t_req": t_req,
            "t_resp": t_resp,
            "tool": tool,
            "args": args,
            "is_error": is_error,
            "result_text": result_text,
            "args_h": sha256_hex(canon(args)),
            "res_h": sha256_hex(canon(result)),
            "prev": self._prev,
        }
        record["h"] = chain_hash(self._prev, record)
        self._f.write(canon(record) + b"\n")
        self._f.flush()
        self._prev = record["h"]
        self.count += 1
        return record

    def close(self):
        self._f.close()


class ChainError(Exception):
    pass


def verify_chain(path):
    """Recompute every h from prev and the record. Returns the list of records.
    Raises ChainError at the first line that does not verify."""
    records = []
    prev = GENESIS
    with open(path, "rb") as f:
        for lineno, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ChainError(f"line {lineno}: not valid JSON ({e})") from None
            if not isinstance(rec, dict) or "h" not in rec or "prev" not in rec:
                raise ChainError(f"line {lineno}: missing fields")
            if rec["prev"] != prev:
                raise ChainError(f"line {lineno}: prev does not match previous record's h")
            body = {k: v for k, v in rec.items() if k != "h"}
            try:
                expected = chain_hash(prev, body)
            except ValueError:
                raise ChainError(f"line {lineno}: prev is not valid hex") from None
            if rec["h"] != expected:
                raise ChainError(f"line {lineno}: h does not match record contents")
            if line.rstrip(b"\n") != canon(rec):
                raise ChainError(f"line {lineno}: line is not in canonical form")
            prev = rec["h"]
            records.append(rec)
    return records
