"""
llm_client.py
=============
Thin client for the local llama.cpp server.

llama-server exposes an OpenAI compatible endpoint at
    http://localhost:8080/v1/chat/completions
so we send a standard chat completion request: a system message (the persona
chosen by the FSM) followed by the running conversation. Only the `requests`
library is used, no vendor SDKs, which keeps the dependency surface small and
makes the data flow easy to inspect.
"""

import json
import time

import requests

# Default endpoint for `llama-server`. Override host/port here if you launch
# the server differently.
SERVER_URL = "http://localhost:8080/v1/chat/completions"

# Network timeout in seconds. Local inference can be slow on first token, so
# this is generous. Latency is one of the things worth measuring in Phase 5.
REQUEST_TIMEOUT = 120

# Fallback replies shared by the blocking and streaming paths, so a streamed
# failure reads exactly like a non-streamed one in the transcript and logs.
FALLBACK_UNREACHABLE = (
    "[The suspect says nothing. The local model server is not "
    "reachable. Start llama-server on port 8080 and try again.]"
)
FALLBACK_TIMEOUT = "[The suspect hesitates too long. The model timed out.]"
FALLBACK_BAD_FORMAT = "[The model returned an unexpected response format.]"


def get_response(system_prompt, messages, temperature=0.7, max_tokens=200,
                 response_format=None, repeat_penalty=1.2, id_slot=None):
    """
    Send one chat completion request to the local server and return the reply.

    Parameters
    ----------
    system_prompt : str
        The persona for this turn, produced by SuspectFSM.get_system_prompt().
    messages : list of dict
        Prior turns as [{"role": "user"|"assistant", "content": str}, ...].
        The system prompt is prepended here, so callers pass only the
        user/assistant exchange.
    temperature : float
        Sampling temperature. Lower is more consistent, higher more varied.
    max_tokens : int
        Upper bound on the reply length.
    response_format : dict, optional
        An OpenAI-style ``response_format`` (e.g. a ``json_schema``) passed
        straight through to llama-server's constrained decoding. The intent
        classifier uses this to force valid, on-spec JSON: the roleplay-tuned
        suspect model otherwise ignores a "you are a classifier" instruction and
        just stays in character, so a grammar constraint is what makes the
        multi-axis classification actually work rather than silently falling back
        to keywords. Left None for ordinary in-character replies.
    repeat_penalty : float
        llama.cpp repetition penalty for in-character replies; without it the
        suspect can lock into repeating one sentence when cornered (1.1 still
        allowed a wall of "I was in the wine cellar" in the Guilty state, hence
        1.2). Applied only
        when ``response_format`` is None: penalising repeats during constrained
        JSON decoding would bias the classifier against giving several axes the
        same (correct) value such as "none".
    id_slot : int, optional
        Pin the request to a specific llama-server slot. The app itself never
        sets this: with ``--parallel 2`` the server's prompt-similarity routing
        already keeps the classifier's fixed prefix and the conversation on
        separate cached slots. It exists so benchmark_llm.py can A/B explicit
        pinning against that automatic routing.

    Returns
    -------
    tuple (str, float)
        (reply, latency_ms). The reply is the assistant's text, or a clear
        fallback message if the server cannot be reached or returns something
        unexpected. latency_ms is the wall clock time the call took, in
        milliseconds, which is logged for the Phase 5 comparison. The fallback
        keeps the UI usable during a demo even if the server is down.
    """
    payload = {
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # llama.cpp extension: reuse the slot's KV cache for the longest common
        # prompt prefix instead of reprocessing from token 0. Recent builds
        # default this on; explicit keeps it true regardless of server version.
        "cache_prompt": True,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    else:
        payload["repeat_penalty"] = repeat_penalty
    if id_slot is not None:
        payload["id_slot"] = id_slot

    start = time.perf_counter()
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        reply = FALLBACK_UNREACHABLE
    except requests.exceptions.Timeout:
        reply = FALLBACK_TIMEOUT
    except (KeyError, IndexError, ValueError):
        reply = FALLBACK_BAD_FORMAT

    latency_ms = (time.perf_counter() - start) * 1000.0
    return reply, latency_ms


# Sentinel returned by parse_sse_line for the end-of-stream marker.
SSE_DONE = object()


def parse_sse_line(line):
    """
    Decode one line of a llama-server SSE stream. Pure, never raises.

    Returns SSE_DONE for the terminal "data: [DONE]" marker, the content delta
    string (possibly "") for a data chunk that carries one, and None for
    everything else: blank keep-alive lines, ": comment" lines, the role-only
    first chunk, the finish_reason chunk, and malformed JSON (skipped rather
    than surfaced -- one bad chunk should not kill an otherwise good reply).
    """
    line = (line or "").strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return SSE_DONE
    try:
        chunk = json.loads(payload)
        delta = chunk["choices"][0].get("delta") or {}
        content = delta.get("content")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None
    return content if isinstance(content, str) else None


def stream_response(system_prompt, messages, temperature=0.7, max_tokens=200,
                    repeat_penalty=1.2, id_slot=None):
    """
    Stream one chat completion from the local server.

    A generator counterpart to get_response() for in-character replies (the
    classifier keeps the blocking call: constrained-JSON output is useless
    until complete). Yields ("delta", text) as tokens arrive, then exactly one
    ("done", info) where info is:

        reply     full accumulated text, stripped -- authoritative: callers
                  must replace any partial display with it (on errors it is a
                  fallback string that never streamed as deltas).
        ttft_ms   start to first content delta, or None if none arrived.
        total_ms  start to stream end (or failure), the generation wall time.
        error     True when reply is a fallback rather than model output.

    Failures anywhere -- before the first token or mid-stream -- collapse to
    the same fallback strings get_response() uses, so a dropped stream reads
    exactly like a failed blocking call. With stream=True the request timeout
    applies per socket read, so a mid-generation stall raises Timeout here
    rather than hanging the turn.
    """
    payload = {
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "cache_prompt": True,
        "repeat_penalty": repeat_penalty,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot

    start = time.perf_counter()
    parts = []
    ttft_ms = None
    error = None
    response = None
    try:
        response = requests.post(
            SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT, stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            piece = parse_sse_line(line)
            if piece is SSE_DONE:
                break
            if piece is None:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            parts.append(piece)
            yield ("delta", piece)
    except requests.exceptions.Timeout:
        error = FALLBACK_TIMEOUT
    except requests.exceptions.RequestException:
        # ConnectionError before the first byte, ChunkedEncodingError on a
        # mid-stream disconnect, HTTPError from raise_for_status: all mean
        # the server failed us, same as unreachable in the blocking client.
        error = FALLBACK_UNREACHABLE
    except (KeyError, IndexError, ValueError):
        error = FALLBACK_BAD_FORMAT
    finally:
        if response is not None:
            response.close()

    total_ms = (time.perf_counter() - start) * 1000.0
    if error is not None:
        info = {"reply": error, "ttft_ms": None, "total_ms": total_ms, "error": True}
    else:
        info = {
            "reply": "".join(parts).strip(),
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "error": False,
        }
    yield ("done", info)


def trim_history(messages, max_turns):
    """
    Return the tail of a chat history, capped at ``max_turns`` question/answer
    pairs plus the trailing user message.

    The full history is kept elsewhere (UI transcript, session logs); this only
    shrinks what is sent to the model, so the prompt stops growing without
    bound and fits a small server context. Load-bearing older facts are
    re-injected via the system prompt (SuspectFSM.get_established_facts), not
    kept as raw transcript.

    The result never starts with an assistant message: Mistral-family chat
    templates expect strict user-first alternation after the system message,
    and a dangling assistant line at the top can degrade or error the template.
    ``max_turns`` <= 0 (or None) disables trimming and returns the list as is.
    """
    if not max_turns or max_turns <= 0 or len(messages) <= 2 * max_turns + 1:
        return messages
    trimmed = messages[-(2 * max_turns + 1):]
    if trimmed and trimmed[0].get("role") == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def server_is_up():
    """
    Quick health check. Returns True if the server answers, False otherwise.
    Useful for showing a status indicator in the UI before the user starts.
    """
    try:
        # The models endpoint is cheap and confirms the server is alive.
        probe = requests.get("http://localhost:8080/v1/models", timeout=3)
        return probe.status_code == 200
    except requests.exceptions.RequestException:
        return False
