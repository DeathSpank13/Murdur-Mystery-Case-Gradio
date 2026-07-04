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

import time

import requests

# Default endpoint for `llama-server`. Override host/port here if you launch
# the server differently.
SERVER_URL = "http://localhost:8080/v1/chat/completions"

# Network timeout in seconds. Local inference can be slow on first token, so
# this is generous. Latency is one of the things worth measuring in Phase 5.
REQUEST_TIMEOUT = 120


def get_response(system_prompt, messages, temperature=0.7, max_tokens=200,
                 response_format=None, repeat_penalty=1.1):
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
        suspect can lock into repeating one sentence when cornered. Applied only
        when ``response_format`` is None: penalising repeats during constrained
        JSON decoding would bias the classifier against giving several axes the
        same (correct) value such as "none".

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
    }
    if response_format is not None:
        payload["response_format"] = response_format
    else:
        payload["repeat_penalty"] = repeat_penalty

    start = time.perf_counter()
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        reply = (
            "[The suspect says nothing. The local model server is not "
            "reachable. Start llama-server on port 8080 and try again.]"
        )
    except requests.exceptions.Timeout:
        reply = "[The suspect hesitates too long. The model timed out.]"
    except (KeyError, IndexError, ValueError):
        reply = "[The model returned an unexpected response format.]"

    latency_ms = (time.perf_counter() - start) * 1000.0
    return reply, latency_ms


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
