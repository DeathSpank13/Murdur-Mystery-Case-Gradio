"""
latency.py
==========
Pacing calibration for the Static control, matched to the Dynamic condition's
real streaming behaviour.

The dynamic condition streams its reply token by token: the player waits
(classifier + time-to-first-token), then watches text appear at the model's
generation pace. The static lookup answers instantly and all at once, which
would give the pre-written condition away immediately. So the static reveal is
synthesized from two quantities observed on real dynamic turns:

  pre_delay_ms    classifier latency + time to first token -- the silent
                  "thinking" wait before any text appears.
  chars_per_sec   reply characters per second of generation after the first
                  token -- the visible typing pace.

  1. ui.py calls record_dynamic_observation() after every live-streamed
     dynamic turn, feeding two rolling windows persisted in
     data/latency_stats.json. The store is module-level (shared across
     sessions in the process) and the file survives restarts, so a session
     where the participant talks to the static detective FIRST still paces
     like the dynamic one did in earlier sessions. On first ever run the
     store bootstraps from dynamic turns in logs/session_*.json that carry a
     ttft_ms field (logs from before streaming existed record only totals,
     which cannot be split into wait + pace, so they contribute nothing).
  2. The static path draws one sample from each window (sample_static_pacing),
     jittered so no two turns pace identically, and reveals its reply through
     reveal_ticks() -- growing prefixes with sleeps between them.

Buffered dynamic turns (nugget-drop turns, which may silently retry) reuse the
same reveal, so live streaming, buffered dynamic and static are one look.

All sampling and chunking functions are pure (no sleep, no I/O when given
samples), so tests cover the maths without waiting on real sleeps.
"""

import glob
import json
import os
import random

# Rolling store of observed dynamic pacing, persisted between runs. The file
# is machine-specific calibration data, not source (gitignored).
STATS_PATH = os.path.join(os.path.dirname(__file__), "data", "latency_stats.json")

# Where to bootstrap samples from when STATS_PATH does not exist yet: the
# dynamic turns of previously recorded study sessions.
LOGS_GLOB = os.path.join(os.path.dirname(__file__), "logs", "session_*.json")

# Keep only the most recent N samples per window: enough to characterise the
# current machine/model, small enough to age out stale history after a
# hardware or model change.
LATENCY_WINDOW = 200

# +/- fraction applied to a drawn sample. Kept small: the real samples already
# carry the distribution's variance, the jitter only prevents exact repeats.
JITTER = 0.15

# Clamp bounds for the synthesized pre-reveal wait. The floor keeps the static
# reply from ever feeling instant; the ceiling sits around the p90 of observed
# classifier+TTFT so static never replays a pathological stall.
PRE_DELAY_MIN_MS = 1000.0
PRE_DELAY_MAX_MS = 15000.0

# Clamp bounds for the synthesized typing pace. Outside this band the reveal
# stops reading as live generation: slower looks broken, faster looks pasted.
PACE_MIN_CPS = 8.0
PACE_MAX_CPS = 120.0

# Accept an observation into the store only inside these ranges: below the
# pre-delay floor is a server-down instant fallback, above the ceiling a
# timeout/retry pathology, and neither should poison the calibration pool.
RECORD_PRE_DELAY_MIN_MS = 300.0
RECORD_PRE_DELAY_MAX_MS = 30000.0
RECORD_PACE_MIN_CPS = 5.0
RECORD_PACE_MAX_CPS = 200.0

# A reply shorter than this gives too noisy a rate estimate to record.
MIN_REPLY_CHARS_FOR_PACE = 20

# Cold-start fallbacks (no samples anywhere): roughly a 2 s classifier plus a
# 1 s first token, and the old fitted 40 ms/char inverted. The wider jitter
# stands in for the spread real samples provide.
FALLBACK_PRE_DELAY_MS = 3000.0
FALLBACK_CHARS_PER_SEC = 25.0
FALLBACK_JITTER = 0.35

# Lazily loaded, then shared for the process lifetime. None = not loaded yet
# (an empty store stays an empty dict-of-lists and is not retried every call).
_store = None


def _valid_pre_delay(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        RECORD_PRE_DELAY_MIN_MS <= value <= RECORD_PRE_DELAY_MAX_MS


def _valid_pace(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        RECORD_PACE_MIN_CPS <= value <= RECORD_PACE_MAX_CPS


def _empty_store():
    return {"pre_delay_ms": [], "chars_per_sec": []}


def _parse_store(raw):
    """
    Pure. Turn whatever was in the stats file into a validated store dict.

    The v1 file was a flat list of total latencies; a total cannot be split
    back into wait + pace, so legacy data is discarded and the fallback
    constants cover the gap until new observations arrive. Anything malformed
    also parses to an empty store -- calibration must never crash a turn.
    """
    store = _empty_store()
    if isinstance(raw, dict):
        pre = raw.get("pre_delay_ms", [])
        pace = raw.get("chars_per_sec", [])
        if isinstance(pre, list):
            store["pre_delay_ms"] = [float(v) for v in pre if _valid_pre_delay(v)]
        if isinstance(pace, list):
            store["chars_per_sec"] = [float(v) for v in pace if _valid_pace(v)]
    return store


def observation_from_turn(turn):
    """
    Pure. Extract a (pre_delay_ms, chars_per_sec) observation from one logged
    turn dict, either value None when it cannot be derived. Only dynamic turns
    logged by the streaming code carry ttft_ms; older logs yield (None, None).
    Shared by the log bootstrap and by tests.
    """
    if turn.get("condition") != "dynamic":
        return None, None
    ttft = turn.get("ttft_ms")
    classifier = turn.get("classifier_latency_ms")
    generation = turn.get("generation_latency_ms")
    reply = turn.get("npc_reply") or ""
    numbers = all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in (ttft, classifier, generation)
    )
    if not numbers:
        return None, None
    pre_delay = classifier + ttft
    pace = None
    if generation > ttft and len(reply) >= MIN_REPLY_CHARS_FOR_PACE:
        pace = len(reply) / ((generation - ttft) / 1000.0)
    return (
        pre_delay if _valid_pre_delay(pre_delay) else None,
        pace if pace is not None and _valid_pace(pace) else None,
    )


def _bootstrap_from_logs():
    """Harvest pacing observations from existing session logs (best effort)."""
    store = _empty_store()
    for path in sorted(glob.glob(LOGS_GLOB)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
            for turn in session.get("turns", []):
                pre_delay, pace = observation_from_turn(turn)
                if pre_delay is not None:
                    store["pre_delay_ms"].append(pre_delay)
                if pace is not None:
                    store["chars_per_sec"].append(pace)
        except Exception:
            continue  # an unreadable log is no reason to fail a turn
    for key in store:
        del store[key][:-LATENCY_WINDOW]
    return store


def _save(store):
    """Best-effort flush; calibration must never crash a study turn."""
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump({"version": 2, **store}, f)
    except Exception:
        pass


def _load_store():
    """Return the shared store dict, loading or bootstrapping it once."""
    global _store
    if _store is None:
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _store = _parse_store(raw)
        except FileNotFoundError:
            _store = _bootstrap_from_logs()
            if _store["pre_delay_ms"] or _store["chars_per_sec"]:
                _save(_store)
        except Exception:
            _store = _empty_store()
    return _store


def record_dynamic_observation(pre_delay_ms, chars_per_sec):
    """
    Feed one live-streamed dynamic turn's pacing into the calibration store.

    Called by ui.py after every live-streamed dynamic reply. Either value may
    be None or out of range independently (server hiccup, reply too short for
    a pace estimate); each is kept only if trustworthy. Buffered drop turns
    are never recorded: their pace was itself synthesized from this store.
    """
    store = _load_store()
    changed = False
    if _valid_pre_delay(pre_delay_ms):
        store["pre_delay_ms"].append(float(pre_delay_ms))
        del store["pre_delay_ms"][:-LATENCY_WINDOW]
        changed = True
    if _valid_pace(chars_per_sec):
        store["chars_per_sec"].append(float(chars_per_sec))
        del store["chars_per_sec"][:-LATENCY_WINDOW]
        changed = True
    if changed:
        _save(store)


def sample_pre_delay_ms(samples=None, rng=random):
    """
    Pick a synthesized pre-reveal wait in milliseconds. Pure: no sleeping, and
    no I/O when `samples` is provided (tests pass a list and a seeded rng).

    With samples: a random observed wait, jittered so repeated draws never
    collide exactly. Without: the cold-start fallback. Clamped to
    [PRE_DELAY_MIN_MS, PRE_DELAY_MAX_MS]; draws over the ceiling land at a
    random point just under it, because a hard clamp would replay the exact
    same maximum wait over and over -- itself a timing tell.
    """
    if samples:
        target = rng.choice(samples)
        target *= rng.uniform(1.0 - JITTER, 1.0 + JITTER)
    else:
        target = FALLBACK_PRE_DELAY_MS * rng.uniform(1.0 - FALLBACK_JITTER, 1.0 + FALLBACK_JITTER)
    if target > PRE_DELAY_MAX_MS:
        target = PRE_DELAY_MAX_MS * rng.uniform(0.9, 1.0)
    return max(PRE_DELAY_MIN_MS, target)


def sample_chars_per_sec(samples=None, rng=random):
    """
    Pick a synthesized typing pace in characters per second. Pure, same shape
    as sample_pre_delay_ms. Soft landings on both bounds (a fixed repeated
    floor pace would be as much of a tell as a fixed ceiling wait).
    """
    if samples:
        target = rng.choice(samples)
        target *= rng.uniform(1.0 - JITTER, 1.0 + JITTER)
    else:
        target = FALLBACK_CHARS_PER_SEC * rng.uniform(1.0 - FALLBACK_JITTER, 1.0 + FALLBACK_JITTER)
    if target > PACE_MAX_CPS:
        target = PACE_MAX_CPS * rng.uniform(0.9, 1.0)
    if target < PACE_MIN_CPS:
        target = PACE_MIN_CPS * rng.uniform(1.0, 1.1)
    return target


def sample_static_pacing(rng=random):
    """Draw the (pre_delay_ms, chars_per_sec) pair for one static reveal."""
    store = _load_store()
    return (
        sample_pre_delay_ms(store["pre_delay_ms"], rng),
        sample_chars_per_sec(store["chars_per_sec"], rng),
    )


def reveal_ticks(reply, chars_per_sec):
    """
    Pure chunking maths for the synthesized typewriter reveal.

    Returns [(prefix, sleep_before_s), ...]: sleep, then show that prefix.
    Chunk size targets ~90 ms per tick at the given pace, held to 2-6 chars so
    slow paces still move visibly and fast paces do not flood the UI with
    yields; the interval preserves the overall rate (total time ~= len/cps).
    The last tick always carries the full reply. An empty reply is a single
    instant tick so callers need no special case.
    """
    reply = reply or ""
    if not reply:
        return [(reply, 0.0)]
    chunk = max(2, min(6, round(chars_per_sec * 0.09)))
    interval = chunk / chars_per_sec
    ticks = []
    for end in range(chunk, len(reply) + chunk, chunk):
        ticks.append((reply[:end], interval))
    return ticks
