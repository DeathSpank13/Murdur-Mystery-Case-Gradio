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
    threshold           an input close to nothing falls back instead of guessing
    empty input         whitespace falls back without touching the index
    fallback cycle      repeated fallbacks rotate through FALLBACK_LINES
    variant rotation    repeat visits walk responses, then cycle repeat lines,
                        with per-entry counters that don't interfere

A data-integrity block validates the real data/suspect_qa.json without any
model: schema rules via load_qa_data, and the three nugget slips (see
nuggets.py) present in their carrier entries' first response so the static
transcript stays consistent with the FSM condition's canon.

A final, *optional* integration check uses the real all-MiniLM-L6-v2 model over
the real data/suspect_qa.json to prove a paraphrase the script never saw still
routes correctly, and sweeps every entry's own example questions to catch
cross-topic collisions. It is skipped (not failed) if sentence-transformers
isn't installed or the model can't be loaded, so the fast suite stays
offline-safe.
"""

import numpy as np
from chromadb.api.types import EmbeddingFunction

import static_dialogue
from nuggets import NUGGETS


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return bool(condition)


# A deterministic stub embedding function: it maps a fixed set of strings to
# orthonormal unit vectors, and any other string to a 4th axis orthogonal to all
# of them. So a stored question matches itself exactly (cosine distance 0) while
# anything off-topic lands at distance 1.0, beyond MATCH_DISTANCE_THRESHOLD.
# Subclasses ChromaDB's EmbeddingFunction so it gets embed_query/embed_documents
# for free (they delegate to __call__); returns numpy arrays as ChromaDB expects.
class StubEmbedding(EmbeddingFunction):
    TABLE = {
        "what is your name?": [1.0, 0.0, 0.0, 0.0],
        "who are you?": [1.0, 0.0, 0.0, 0.0],
        "where were you?": [0.0, 1.0, 0.0, 0.0],
        "what is your alibi?": [0.0, 1.0, 0.0, 0.0],
        "did you kill him?": [0.0, 0.0, 1.0, 0.0],
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
# walk), alibi has a single response with repeats, and deny has variants but no
# repeat lines, covering the wrap-around branch of _select_variant.
STUB_QA = [
    {"id": "identity",
     "responses": ["I am Eleanor Vance.", "Still Eleanor Vance."],
     "repeat_responses": ["We have established my name."],
     "questions": ["What is your name?", "Who are you?"]},
    {"id": "alibi",
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
            "off-topic input falls back instead of guessing",
            static_dialogue.get_response("tell me about the weather") in static_dialogue.FALLBACK_LINES,
        ))
        results.append(check(
            "empty input falls back",
            static_dialogue.get_response("   ") in static_dialogue.FALLBACK_LINES,
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


# Which QA entry's first response carries each slip. The drop markers
# themselves come from nuggets.py, so the FSM condition stays the single
# source of truth for the slip wording.
SLIP_CARRIERS = {"wound": "weapon", "corridor": "lastseen", "cut": "wellbeing"}


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
    for nugget_id, entry_id in SLIP_CARRIERS.items():
        entry = by_id.get(entry_id)
        markers = NUGGETS[nugget_id]["drop_markers"]
        first = entry["responses"][0].lower() if entry else ""
        results.append(check(
            f"slip '{nugget_id}' markers {markers} baked into {entry_id}.responses[0]",
            entry is not None and all(marker in first for marker in markers),
        ))
        # The slip should drop exactly once, like the dynamic condition: no
        # marker may leak into the carrier's later variants or repeat lines.
        later = (entry["responses"][1:] + entry.get("repeat_responses", [])) if entry else []
        results.append(check(
            f"slip '{nugget_id}' does not repeat in later {entry_id} lines",
            entry is not None and not any(
                marker in line.lower() for line in later for marker in markers
            ),
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

        # Self-retrieval sweep: every entry's own example questions must route
        # back to that entry. Catches cross-topic collisions when neighbouring
        # topics (alibi/wine_cellar, weapon/fingerprints, ...) drift too close
        # in embedding space; fix failures by rewording the example question,
        # not by moving the threshold.
        collection = static_dialogue._get_collection()
        collisions = []
        for entry in data:
            for question in entry["questions"]:
                result = collection.query(query_texts=[question], n_results=1)
                matched = result["metadatas"][0][0]["entry_id"]
                if matched != entry["id"]:
                    collisions.append(f"{entry['id']!r}: {question!r} -> {matched!r}")
        for line in collisions:
            print(f"       collision {line}")
        results.append(check(
            f"self-retrieval sweep: every example question routes home "
            f"({len(collisions)} collisions)",
            not collisions,
        ))
        return results
    except Exception as exc:  # model download/load failure shouldn't fail the suite
        print(f"[SKIP] integration check (model unavailable: {exc})")
        return []
    finally:
        static_dialogue._collection = None


def run():
    results = run_fast()
    results += run_data_integrity()
    results += run_integration()
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
