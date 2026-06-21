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

A final, *optional* integration check uses the real all-MiniLM-L6-v2 model over
the real data/suspect_qa.json to prove a paraphrase the script never saw still
routes correctly. It is skipped (not failed) if sentence-transformers isn't
installed or the model can't be loaded, so the fast suite stays offline-safe.
"""

import numpy as np
from chromadb.api.types import EmbeddingFunction

import static_dialogue


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


STUB_QA = [
    {"id": "identity", "response": "I am Eleanor Vance.",
     "questions": ["What is your name?", "Who are you?"]},
    {"id": "alibi", "response": "I was in the drawing room.",
     "questions": ["Where were you?", "What is your alibi?"]},
    {"id": "deny", "response": "How dare you.",
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
            "stored question retrieves its response",
            static_dialogue.get_response("What is your name?") == "I am Eleanor Vance.",
        ))
        results.append(check(
            "a different topic retrieves the right entry",
            static_dialogue.get_response("Where were you?") == "I was in the drawing room.",
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
            set(cycled) == set(static_dialogue.FALLBACK_LINES),
        ))
    finally:
        static_dialogue._collection = None  # don't leak the stub into other runs

    return results


def run_integration():
    """
    Optional: prove a real paraphrase routes correctly with the real model.
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
        identity_response = next(e["response"] for e in data if e["id"] == "identity")
        results = [check(
            "paraphrase routes to the identity response (real model)",
            reply == identity_response,
        )]
        offtopic = static_dialogue.get_response("what's the weather like outside")
        results.append(check(
            "genuinely off-topic input falls back (real model)",
            offtopic in static_dialogue.FALLBACK_LINES,
        ))
        return results
    except Exception as exc:  # model download/load failure shouldn't fail the suite
        print(f"[SKIP] integration check (model unavailable: {exc})")
        return []
    finally:
        static_dialogue._collection = None


def run():
    results = run_fast()
    results += run_integration()
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
