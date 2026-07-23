"""
test_llm_stream.py
==================
Offline checks for llm_client's SSE streaming path. No test framework and no
server: run `python test_llm_stream.py` and it prints a pass or fail line per
check.

parse_sse_line is pure, so its cases are fed as literal strings. stream_response
is exercised by monkeypatching requests.post with a canned response object
whose iter_lines() replays a recorded stream (or raises mid-way), proving the
event protocol -- ("delta", ...) events then exactly one ("done", info) -- and
that every failure mode collapses to the same fallback strings the blocking
client uses.
"""

import json

import requests

import llm_client
from llm_client import SSE_DONE, parse_sse_line


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


def _chunk(content=None, role=None, finish_reason=None):
    """Build one llama-server style SSE data line."""
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return "data: " + json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    )


class _FakeResponse:
    """Stands in for requests.post(..., stream=True)'s return value."""

    def __init__(self, lines=(), raise_after=None, exc=None):
        self._lines = list(lines)
        self._raise_after = raise_after  # raise exc after this many lines
        self._exc = exc
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for index, line in enumerate(self._lines):
            if self._raise_after is not None and index >= self._raise_after:
                raise self._exc
            yield line
        if self._raise_after is not None and self._raise_after >= len(self._lines):
            raise self._exc

    def close(self):
        self.closed = True


def _run_stream(fake_or_exc):
    """Drive stream_response against a patched requests.post, restore after."""
    original = requests.post

    def fake_post(*args, **kwargs):
        if isinstance(fake_or_exc, Exception):
            raise fake_or_exc
        return fake_or_exc

    requests.post = fake_post
    try:
        return list(llm_client.stream_response("system prompt", []))
    finally:
        requests.post = original


def run_parse():
    results = []
    results.append(check(
        "content delta decodes to its text",
        parse_sse_line(_chunk(content="Hel")) == "Hel",
    ))
    results.append(check(
        "empty-string delta decodes to '' (not dropped as None)",
        parse_sse_line(_chunk(content="")) == "",
    ))
    results.append(check(
        "terminal [DONE] marker maps to the sentinel",
        parse_sse_line("data: [DONE]") is SSE_DONE
        and parse_sse_line("data:[DONE]") is SSE_DONE,
    ))
    results.append(check(
        "keep-alive blank line, comment line and None are ignored",
        parse_sse_line("") is None
        and parse_sse_line(": keep-alive") is None
        and parse_sse_line(None) is None,
    ))
    results.append(check(
        "role-only first chunk and finish-reason chunk are ignored",
        parse_sse_line(_chunk(role="assistant")) is None
        and parse_sse_line(_chunk(finish_reason="stop")) is None,
    ))
    results.append(check(
        "malformed JSON and shape surprises are skipped, never raised",
        parse_sse_line("data: {oops") is None
        and parse_sse_line('data: {"choices": []}') is None
        and parse_sse_line('data: {"choices": [{"delta": null}]}') is None
        and parse_sse_line('data: {"choices": [{"delta": {"content": 5}}]}') is None,
    ))
    return results


def run_stream():
    results = []

    # Happy path: deltas stream in order, then one done event with the
    # stripped accumulated reply and both timings measured.
    fake = _FakeResponse(lines=[
        _chunk(role="assistant"),
        _chunk(content=" I was"),
        "",
        _chunk(content=" in the"),
        _chunk(content=" cellar. "),
        _chunk(finish_reason="stop"),
        "data: [DONE]",
    ])
    events = _run_stream(fake)
    deltas = [value for kind, value in events if kind == "delta"]
    done = [value for kind, value in events if kind == "done"]
    info = done[0] if done else {}
    results.append(check(
        "happy path: delta events in order, exactly one done event last",
        deltas == [" I was", " in the", " cellar. "]
        and len(done) == 1 and events[-1][0] == "done",
    ))
    results.append(check(
        "happy path: reply is the stripped accumulation, no error",
        info.get("reply") == "I was in the cellar."
        and info.get("error") is False
        and isinstance(info.get("ttft_ms"), float)
        and isinstance(info.get("total_ms"), float)
        and info["ttft_ms"] <= info["total_ms"],
    ))
    results.append(check(
        "happy path: the response is closed after the stream",
        fake.closed,
    ))

    # Server unreachable before the first byte: one done event carrying the
    # same fallback string the blocking client uses.
    events = _run_stream(requests.exceptions.ConnectionError())
    results.append(check(
        "connection error: single done event with the unreachable fallback",
        events == [("done", events[0][1])]
        and events[0][1]["reply"] == llm_client.FALLBACK_UNREACHABLE
        and events[0][1]["error"] is True
        and events[0][1]["ttft_ms"] is None,
    ))

    # Mid-stream stall: partial deltas were already yielded, then the done
    # event replaces them with the timeout fallback.
    fake = _FakeResponse(
        lines=[_chunk(content="I was")],
        raise_after=1,
        exc=requests.exceptions.ReadTimeout(),
    )
    events = _run_stream(fake)
    info = events[-1][1]
    results.append(check(
        "mid-stream timeout: deltas first, then done with the timeout fallback",
        events[0] == ("delta", "I was")
        and events[-1][0] == "done"
        and info["reply"] == llm_client.FALLBACK_TIMEOUT
        and info["error"] is True
        and fake.closed,
    ))

    # Mid-stream disconnect (chunked encoding error is a RequestException):
    # same collapse, unreachable fallback.
    fake = _FakeResponse(
        lines=[_chunk(content="I was")],
        raise_after=1,
        exc=requests.exceptions.ChunkedEncodingError(),
    )
    events = _run_stream(fake)
    results.append(check(
        "mid-stream disconnect: done event with the unreachable fallback",
        events[-1][1]["reply"] == llm_client.FALLBACK_UNREACHABLE
        and events[-1][1]["error"] is True,
    ))

    # A stream that ends without [DONE] (server closed cleanly early) still
    # produces a normal done event from whatever arrived.
    fake = _FakeResponse(lines=[_chunk(content="Hm.")])
    events = _run_stream(fake)
    results.append(check(
        "stream ending without [DONE] still finalises the reply",
        events[-1][1]["reply"] == "Hm." and events[-1][1]["error"] is False,
    ))
    return results


def run():
    results = run_parse() + run_stream()
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
