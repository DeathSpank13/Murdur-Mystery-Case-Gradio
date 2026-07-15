"""
latency.py
==========
Simulated "thinking" latency for the Static control, calibrated to the Dynamic
condition's real response times.

The static lookup answers almost instantly, which would let testers spot the
pre-written condition by speed alone (and confound the study). Earlier versions
slept for a fixed length-scaled formula, but its 6 s ceiling sat *below* the
dynamic condition's median (~9 s on the study machine), so timing was still a
tell. Now the delay is sampled from the dynamic condition's actually observed
latencies:

  1. ui.py calls record_dynamic_latency() after every dynamic turn, feeding a
     rolling window of real latencies that persists in data/latency_stats.json.
     The store is module-level (shared across sessions in the process) and the
     file survives restarts, so a session where the participant talks to the
     static detective FIRST still waits like the dynamic one did in earlier
     sessions. On first ever run the store bootstraps from the dynamic turns
     already recorded in logs/session_*.json.
  2. simulate_latency() draws a random observed sample, applies small jitter so
     no two waits are identical, clamps, and sleeps. With no samples anywhere
     (fresh checkout, no logs) it falls back to a length-scaled formula whose
     constants were fitted to the observed distribution.

sample_latency_ms() is pure (no sleep, no I/O when given samples), so tests can
cover the maths without waiting on real sleeps.
"""

import glob
import json
import os
import random
import time

# Rolling store of observed dynamic latencies, persisted between runs. The
# file is machine-specific calibration data, not source (gitignored).
STATS_PATH = os.path.join(os.path.dirname(__file__), "data", "latency_stats.json")

# Where to bootstrap samples from when STATS_PATH does not exist yet: the
# dynamic turns of previously recorded study sessions.
LOGS_GLOB = os.path.join(os.path.dirname(__file__), "logs", "session_*.json")

# Keep only the most recent N samples: enough to characterise the current
# machine/model, small enough to age out stale history after a hardware or
# model change.
LATENCY_WINDOW = 200

# +/- fraction applied to a drawn sample. Kept small: the real samples already
# carry the distribution's variance, the jitter only prevents exact repeats.
LATENCY_JITTER = 0.15

# Clamp bounds for the simulated wait. The floor keeps replies from ever
# feeling instant (observed dynamic minimum is ~4 s); the ceiling sits around
# the p85-p90 of observed latencies so the static condition matches the bulk
# of the distribution without replaying pathological retry turns.
SIM_LATENCY_MIN_MS = 1500.0
SIM_LATENCY_MAX_MS = 20000.0

# Accept a dynamic sample into the store only inside this range: below it is a
# server-down instant fallback reply, above it a timeout/retry pathology, and
# neither should poison the calibration pool.
RECORD_MIN_MS = 1000.0
RECORD_MAX_MS = 60000.0

# Cold-start fallback (no samples anywhere): base + per-char, fitted to the
# observed distribution -- a typical ~150-char reply lands near the ~9 s
# median, and the wider jitter stands in for the spread real samples provide.
FALLBACK_BASE_MS = 4000.0
FALLBACK_PER_CHAR_MS = 40.0
FALLBACK_JITTER = 0.35

# Lazily loaded, then shared for the process lifetime. None = not loaded yet
# (an empty store stays an empty list and is not retried every call).
_samples = None


def _valid_sample(latency_ms):
    """True if a recorded dynamic latency is trustworthy calibration data."""
    return (
        isinstance(latency_ms, (int, float))
        and RECORD_MIN_MS <= latency_ms <= RECORD_MAX_MS
    )


def _bootstrap_from_logs():
    """Harvest dynamic-turn latencies from existing session logs (best effort)."""
    harvested = []
    for path in sorted(glob.glob(LOGS_GLOB)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
            for turn in session.get("turns", []):
                if turn.get("condition") == "dynamic" and _valid_sample(turn.get("latency_ms")):
                    harvested.append(float(turn["latency_ms"]))
        except Exception:
            continue  # an unreadable log is no reason to fail a turn
    return harvested[-LATENCY_WINDOW:]


def _save(samples):
    """Best-effort flush; calibration must never crash a study turn."""
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(samples, f)
    except Exception:
        pass


def _load_store():
    """Return the shared sample list, loading or bootstrapping it once."""
    global _samples
    if _samples is None:
        loaded = []
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded = [float(s) for s in raw if _valid_sample(s)]
        except FileNotFoundError:
            loaded = _bootstrap_from_logs()
            if loaded:
                _save(loaded)
        except Exception:
            loaded = []
        _samples = loaded[-LATENCY_WINDOW:]
    return _samples


def record_dynamic_latency(latency_ms):
    """
    Feed one observed dynamic-turn latency into the calibration store.

    Called by ui.py after every dynamic reply. Out-of-range values (server
    down, timeout retries) are silently dropped.
    """
    if not _valid_sample(latency_ms):
        return
    samples = _load_store()
    samples.append(float(latency_ms))
    del samples[:-LATENCY_WINDOW]
    _save(samples)


def sample_latency_ms(reply, samples=None, rng=random):
    """
    Pick a simulated latency in milliseconds. Pure: no sleeping, and no I/O
    when `samples` is provided (tests pass a list and a seeded rng).

    With samples: a random observed dynamic latency, jittered so repeated
    draws never collide exactly. Without: the fitted length-scaled formula.
    Always clamped to [SIM_LATENCY_MIN_MS, SIM_LATENCY_MAX_MS]; draws over the
    ceiling land at a random point just under it, because the dynamic tail is
    heavy enough that a hard clamp would replay the exact same maximum wait
    over and over -- itself a timing tell.
    """
    if samples:
        target = rng.choice(samples)
        target *= rng.uniform(1.0 - LATENCY_JITTER, 1.0 + LATENCY_JITTER)
    else:
        target = FALLBACK_BASE_MS + FALLBACK_PER_CHAR_MS * len(reply or "")
        target *= rng.uniform(1.0 - FALLBACK_JITTER, 1.0 + FALLBACK_JITTER)
    if target > SIM_LATENCY_MAX_MS:
        target = SIM_LATENCY_MAX_MS * rng.uniform(0.9, 1.0)
    return max(SIM_LATENCY_MIN_MS, target)


def simulate_latency(reply):
    """
    Sleep for a dynamic-condition-plausible delay, then return the actual time
    slept in milliseconds. Used only by the static control (see ui.py, where a
    dev toggle can skip it entirely).
    """
    target_ms = sample_latency_ms(reply, _load_store())
    start = time.perf_counter()
    time.sleep(target_ms / 1000.0)
    return (time.perf_counter() - start) * 1000.0
