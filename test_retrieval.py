"""
test_retrieval.py
=================
Lightweight checks for the Static-mode semantic retrieval (static_dialogue.py).
No test framework needed: run `python test_retrieval.py` and it prints a pass or
fail line per check.

The fast checks inject a tiny deterministic *stub* embedding function and a
3-entry database, so they need only ChromaDB installed — no Sentence Transformer
model and no network download. They guard the retrieval mechanics that matter:

    nearest neighbour   a stored question routes to its entry's response
    margin / bands      an ambiguous tie between two entries clarifies instead
                        of guessing; a near-miss clarifies naming the topic; a
                        clear miss falls back
    empty input         whitespace falls back without touching the index
    clarify/fallback    repeated clarifies/fallbacks rotate through their lines
    variant rotation    repeat visits walk responses, then cycle repeat lines,
                        with per-entry counters that don't interfere

A data-integrity block validates the real data/suspect_qa.json without any
model: schema rules via load_qa_data, the three nugget slips (see nuggets.py)
present in their carrier entries' first response — and nowhere else — so the
static transcript stays consistent with the FSM condition's canon, and an
authored topic_hint on every entry.

A latency block unit-tests the pure sampling maths in latency.py (no sleeps).

A final, *optional* integration check uses the real all-MiniLM-L6-v2 model over
the real data/suspect_qa.json to prove a paraphrase the script never saw still
routes correctly, and sweeps every entry's own example questions to confirm
each routes home *confidently* (band "match", not merely top-1-correct), which
catches both cross-topic collisions and margin regressions when new questions
shift the embedding neighbourhoods. It is skipped (not failed) if
sentence-transformers isn't installed or the model can't be loaded, so the fast
suite stays offline-safe.
"""

import random

import numpy as np
from chromadb.api.types import EmbeddingFunction

import latency
import static_dialogue
from nuggets import NUGGETS


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


# A deterministic stub embedding function: it maps a fixed set of strings to
# unit vectors, and any other string to a 4th axis orthogonal to the three
# entry axes. A stored question matches itself exactly (cosine distance 0),
# anything off-topic lands at distance 1.0 (beyond CLARIFY_DISTANCE_THRESHOLD),
# and two crafted inputs land in the clarify band: one nearly equidistant
# between the identity and alibi axes (margin too small), one at distance 0.6
# from alibi (between the match and clarify thresholds).
# Subclasses ChromaDB's EmbeddingFunction so it gets embed_query/embed_documents
# for free (they delegate to __call__); returns numpy arrays as ChromaDB expects.
class StubEmbedding(EmbeddingFunction):
    # cos to identity axis ~0.717 (distance ~0.283), to alibi axis ~0.697
    # (distance ~0.303): both inside the match threshold but only 0.02 apart,
    # far below MATCH_MARGIN = 0.08.
    _AMBIGUOUS = [0.717, 0.697, 0.0, 0.0]
    # cos to alibi axis 0.4 (distance 0.6): above MATCH_DISTANCE_THRESHOLD =
    # 0.55, below CLARIFY_DISTANCE_THRESHOLD = 0.7. The 4th-axis component
    # only pads the norm; no stored question lies along it.
    _NEAR_MISS = [0.0, 0.4, 0.0, 0.9165151389911680]

    TABLE = {
        "what is your name?": [1.0, 0.0, 0.0, 0.0],
        "who are you?": [1.0, 0.0, 0.0, 0.0],
        "where were you?": [0.0, 1.0, 0.0, 0.0],
        "what is your alibi?": [0.0, 1.0, 0.0, 0.0],
        "did you kill him?": [0.0, 0.0, 1.0, 0.0],
        "is this about your name or your alibi?": _AMBIGUOUS,
        "something vaguely about where you were": _NEAR_MISS,
    }
    DEFAULT = [0.0, 0.0, 0.0, 1.0]

    def __init__(self):
        pass

    def __call__(self, input):
        return [
            np.array(self.TABLE.get(t.strip().lower(), self.DEFAULT), dtype=np.float32)
            for t in input
        ]

    @staticmethod
    def name():
        return "stub"


# Mirrors the real schema: identity has variants AND repeat lines (the full
# walk), alibi has a single response with repeats plus an authored topic_hint,
# and deny has variants but no repeat lines, covering the wrap-around branch of
# _select_variant. identity has NO topic_hint, covering the id-derived default.
STUB_QA = [
    {"id": "identity",
     "responses": ["I am Eleanor Vance.", "Still Eleanor Vance."],
     "repeat_responses": ["We have established my name."],
     "questions": ["What is your name?", "Who are you?"]},
    {"id": "alibi",
     "topic_hint": "my whereabouts",
     "responses": ["I was in the drawing room."],
     "repeat_responses": ["Asked and answered.", "Still the drawing room."],
     "questions": ["Where were you?", "What is your alibi?"]},
    {"id": "deny",
     "responses": ["How dare you.", "I did not harm him."],
     "questions": ["Did you kill him?"]},
]


def run_fast():
    """Retrieval mechanics against the stub embedder. Resets module state after."""
    results = []

    # Inject the stub-built collection so get_response() exercises the real
    # query/threshold/fallback path without loading the real model.
    static_dialogue._collection = static_dialogue._build_collection(
        STUB_QA, StubEmbedding()
    )
    try:
        results.append(check(
            "stored question retrieves its first response (stateless call)",
            static_dialogue.get_response("What is your name?") == "I am Eleanor Vance.",
        ))
        results.append(check(
            "a different topic retrieves the right entry",
            static_dialogue.get_response("Where were you?") == "I was in the drawing room.",
        ))
        results.append(check(
            "stateless calls never advance the rotation",
            static_dialogue.get_response("What is your name?") == "I am Eleanor Vance.",
        ))
        results.append(check(
            "off-topic input (distance 1.0, past the clarify band) falls back",
            static_dialogue.get_response("tell me about the weather") in static_dialogue.FALLBACK_LINES,
        ))
        results.append(check(
            "empty input falls back",
            static_dialogue.get_response("   ") in static_dialogue.FALLBACK_LINES,
        ))

        # Band routing. A stored question is a confident match even though its
        # entry owns several of the top-k neighbours (corroboration, not
        # ambiguity): the margin is measured against the best *different* entry.
        band, metadata = static_dialogue._route("What is your name?")
        results.append(check(
            "stored question routes to band 'match' despite same-entry top hits",
            band == "match" and metadata["entry_id"] == "identity",
        ))

        # Nearly equidistant between identity and alibi: margin too small, so
        # she asks for clarification naming the nearest topic (identity has no
        # authored hint, so the id-derived default is used).
        results.append(check(
            "ambiguous tie between entries clarifies with the id-derived hint",
            static_dialogue.get_response("is this about your name or your alibi?")
            == static_dialogue.CLARIFY_TEMPLATES[0].format(topic="identity"),
        ))

        # Distance 0.6 from alibi: too far for a match, close enough to name
        # the topic, using the authored topic_hint.
        results.append(check(
            "near-miss clarifies naming the authored topic_hint",
            static_dialogue.get_response("something vaguely about where you were")
            == static_dialogue.CLARIFY_TEMPLATES[0].format(topic="my whereabouts"),
        ))

        # Repeated clarifies rotate templates under the reserved "_clarify" key.
        counts = {}
        clarified = [
            static_dialogue.get_response("is this about your name or your alibi?", counts)
            for _ in range(len(static_dialogue.CLARIFY_TEMPLATES))
        ]
        results.append(check(
            "repeated clarifies cycle through all templates",
            clarified == [t.format(topic="identity") for t in static_dialogue.CLARIFY_TEMPLATES],
        ))
        results.append(check(
            "clarify counter is tracked under '_clarify' without touching entries",
            counts == {"_clarify": len(static_dialogue.CLARIFY_TEMPLATES)},
        ))

        # Repeated fallbacks rotate through the list rather than repeating one.
        counts = {}
        cycled = [static_dialogue.get_response("xyzzy", counts) for _ in range(len(static_dialogue.FALLBACK_LINES))]
        results.append(check(
            "repeated fallbacks cycle through all lines",
            cycled == static_dialogue.FALLBACK_LINES,
        ))

        # Variant rotation: responses walk in order, then repeat lines cycle.
        counts = {}
        walk = [static_dialogue.get_response("What is your name?", counts) for _ in range(5)]
        results.append(check(
            "repeat visits walk the variants in order",
            walk[:2] == ["I am Eleanor Vance.", "Still Eleanor Vance."],
        ))
        results.append(check(
            "exhausted variants cycle the repeat lines",
            walk[2:] == ["We have established my name."] * 3,
        ))

        # An entry with no repeat lines wraps back through its variants.
        counts = {}
        wrap = [static_dialogue.get_response("Did you kill him?", counts) for _ in range(4)]
        results.append(check(
            "entries without repeat lines wrap their variants",
            wrap == ["How dare you.", "I did not harm him."] * 2,
        ))

        # Counters are per entry: visiting one topic must not advance another,
        # and the fallback counter must not collide with entry counters.
        counts = {}
        static_dialogue.get_response("What is your name?", counts)
        static_dialogue.get_response("xyzzy", counts)
        results.append(check(
            "per-entry counters don't interfere (topic, fallback, other topic)",
            static_dialogue.get_response("Where were you?", counts) == "I was in the drawing room."
            and counts == {"identity": 1, "_fallback": 1, "alibi": 1},
        ))
    finally:
        static_dialogue._collection = None  # don't leak the stub into other runs

    return results


# Which QA entries' first responses carry each slip. The drop markers
# themselves come from nuggets.py, so the FSM condition stays the single
# source of truth for the slip wording. "cut" has two carriers: the thumb
# story legitimately appears in both the wellbeing and wine_cellar canonical
# answers (mirroring nuggets.py, whose cut topic covers both subjects).
SLIP_CARRIERS = {
    "wound": ["weapon"],
    "corridor": ["lastseen"],
    "cut": ["wellbeing", "wine_cellar"],
}


def run_data_integrity():
    """Validate the real database without loading any model."""
    results = []
    try:
        data = static_dialogue.load_qa_data()
        results.append(check("real database passes schema validation", True))
    except Exception as exc:
        results.append(check(f"real database passes schema validation ({exc})", False))
        return results

    by_id = {entry["id"]: entry for entry in data}
    for nugget_id, carrier_ids in SLIP_CARRIERS.items():
        markers = NUGGETS[nugget_id]["drop_markers"]
        for entry_id in carrier_ids:
            entry = by_id.get(entry_id)
            first = entry["responses"][0].lower() if entry else ""
            results.append(check(
                f"slip '{nugget_id}' markers {markers} baked into {entry_id}.responses[0]",
                entry is not None and all(marker in first for marker in markers),
            ))

    # Global leak scan: a slip should drop exactly once per topic, like the
    # dynamic condition, so no marker may appear in ANY spoken line other than
    # its carriers' first responses -- not in later variants or repeat lines
    # of the carriers, and not anywhere in any other entry. (Questions are the
    # player's words, not Eleanor's, so they are not scanned.) The dynamic
    # marker check (ui._marker_in_reply) fires on any single marker, so any
    # single marker counts as a leak here too.
    leaks = []
    for entry in data:
        for nugget_id, carrier_ids in SLIP_CARRIERS.items():
            markers = NUGGETS[nugget_id]["drop_markers"]
            spoken = entry["responses"] + entry.get("repeat_responses", [])
            if entry["id"] in carrier_ids:
                spoken = spoken[1:]  # responses[0] is the sanctioned drop
            for line in spoken:
                for marker in markers:
                    if marker in line.lower():
                        leaks.append(f"'{marker}' in {entry['id']}: {line[:60]!r}")
    for leak in leaks:
        print(f"       leak {leak}")
    results.append(check(
        f"no slip marker leaks outside carrier first responses ({len(leaks)} leaks)",
        not leaks,
    ))

    # Every entry should have an authored topic_hint so the clarify band never
    # falls back to a raw id-derived phrase in the real study.
    missing = [entry["id"] for entry in data if not entry.get("topic_hint")]
    results.append(check(
        f"every entry has an authored topic_hint ({len(missing)} missing)",
        not missing,
    ))
    return results


def run_integration():
    """
    Optional: prove real paraphrases route correctly with the real model.
    Returns a list of results, or [] if the model stack isn't available.
    """
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        print("[SKIP] integration check (sentence-transformers not installed)")
        return []

    try:
        static_dialogue._collection = None  # force a real build over real data
        # A paraphrase that appears in no example question verbatim.
        reply = static_dialogue.get_response("remind me what you're called")
        data = static_dialogue.load_qa_data()
        identity_first = next(e["responses"][0] for e in data if e["id"] == "identity")
        results = [check(
            "paraphrase routes to the identity response (real model)",
            reply == identity_first,
        )]
        offtopic = static_dialogue.get_response("what's the weather like outside")
        results.append(check(
            "genuinely off-topic input falls back (real model)",
            offtopic in static_dialogue.FALLBACK_LINES,
        ))

        # Confident-band sweep: every entry's own example questions must route
        # home through _route with band "match" -- not merely top-1-correct.
        # Catches cross-topic collisions when neighbouring topics
        # (alibi/wine_cellar, weapon/fingerprints, ...) drift too close in
        # embedding space, AND margin regressions where a new question pulls
        # two entries within MATCH_MARGIN of each other. Fix failures by
        # rewording the offending example question, not by moving thresholds.
        misroutes = []
        for entry in data:
            for question in entry["questions"]:
                band, metadata = static_dialogue._route(question)
                matched = metadata["entry_id"] if metadata else None
                if band != "match" or matched != entry["id"]:
                    misroutes.append(
                        f"{entry['id']!r}: {question!r} -> band={band!r}, {matched!r}"
                    )
        for line in misroutes:
            print(f"       misroute {line}")
        results.append(check(
            f"confident-band sweep: every example question routes home as a "
            f"match ({len(misroutes)} misroutes)",
            not misroutes,
        ))
        return results
    except Exception as exc:  # model download/load failure shouldn't fail the suite
        print(f"[SKIP] integration check (model unavailable: {exc})")
        return []
    finally:
        static_dialogue._collection = None


def run_latency():
    """Unit-test the pure latency sampling maths (no sleeps, seeded rng)."""
    results = []
    rng = random.Random(0)

    # Empirical path: a drawn sample stays within jitter of the input sample.
    lo = 8000.0 * (1.0 - latency.LATENCY_JITTER)
    hi = 8000.0 * (1.0 + latency.LATENCY_JITTER)
    drawn = [latency.sample_latency_ms("x" * 100, [8000.0], rng) for _ in range(20)]
    results.append(check(
        "empirical sample stays within jitter of the observed latency",
        all(lo <= value <= hi for value in drawn),
    ))

    # Clamping at both ends: a tiny observed sample can't produce an instant
    # reply, a huge one can't stall past the ceiling.
    results.append(check(
        "small samples clamp to the floor",
        latency.sample_latency_ms("reply", [500.0], rng) == latency.SIM_LATENCY_MIN_MS,
    ))
    over = [latency.sample_latency_ms("reply", [50000.0], rng) for _ in range(20)]
    results.append(check(
        "large samples land just under the ceiling, never identically at it",
        all(0.9 * latency.SIM_LATENCY_MAX_MS <= value <= latency.SIM_LATENCY_MAX_MS
            for value in over)
        and len(set(over)) > 1,
    ))

    # Cold-start fallback (no samples anywhere): scales with reply length and
    # respects the same clamps.
    short = [latency.sample_latency_ms("", [], rng) for _ in range(20)]
    long = [latency.sample_latency_ms("x" * 400, [], rng) for _ in range(20)]
    base_lo = latency.FALLBACK_BASE_MS * (1.0 - latency.FALLBACK_JITTER)
    base_hi = latency.FALLBACK_BASE_MS * (1.0 + latency.FALLBACK_JITTER)
    results.append(check(
        "cold-start formula: empty reply lands around the base delay",
        all(base_lo <= value <= base_hi for value in short),
    ))
    results.append(check(
        "cold-start formula: a long reply waits longer, up to the ceiling",
        all(value > base_hi and value <= latency.SIM_LATENCY_MAX_MS for value in long),
    ))

    # Record-range guard: server-down instant replies and timeout pathologies
    # must never enter the calibration pool.
    results.append(check(
        "record validation rejects outliers and accepts normal turns",
        not latency._valid_sample(200.0)
        and not latency._valid_sample(120000.0)
        and not latency._valid_sample(None)
        and latency._valid_sample(9000.0),
    ))
    return results


def run():
    results = run_fast()
    results += run_data_integrity()
    results += run_latency()
    results += run_integration()
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
