"""
static_dialogue.py
==================
The model-free "Static Control" version of the interrogation, now built as a
semantic retrieval lookup. This is still the non-AI baseline that the dynamic
FSM plus LLM version is compared against in the user study: nothing is
generated, every reply is a fixed, pre-written line pulled verbatim from a
database.

How it works:

  1. A set of existing dialogue entries lives in data/suspect_qa.json. Each
     entry pairs an ordered list of pre-written response variants (plus
     optional "you've asked me this before" repeat lines) with several
     natural-language example questions that should map to it. The example
     questions are embedded once with a Sentence Transformer
     (all-MiniLM-L6-v2) and stored in a ChromaDB collection.
  2. When the player writes something, that text is embedded with the same
     model and ChromaDB returns the nearest neighbour by cosine distance.
  3. The matched entry's next unspoken variant is returned: the first ask gets
     responses[0] (the canonical answer), later asks walk the remaining
     variants in order, and once those run out the repeat_responses cycle. The
     per-session visit_counts dict passed in by ui.py tracks the position, so
     the rotation is deterministic and resets with the session. If nothing is
     close enough (distance above MATCH_DISTANCE_THRESHOLD), a generic
     fallback line is returned instead, cycling from polite to testy.

This replaces the older best-match *keyword* tree. The win over keywords is
that paraphrases the script never anticipated ("remind me what you're called")
still route to the right answer by meaning rather than by literal word overlap.
It stays squarely the simple control, though: every line is pre-written and the
suspect never confesses. The three slips from nuggets.py (neck / doorway at a
quarter to ten / cut thumb) are baked into the first response of their carrier
entries as fixed text, so the mystery is technically solvable from a static
transcript too -- but there are no confrontation or confession mechanics here;
the guilt fact itself lives only in the dynamic FSM overlays (see fsm.py). The
point of the comparison is to show what is gained, and what is lost, when these
fixed lines are replaced by a model whose persona is steered by the FSM.

The embedding model and the ChromaDB collection are heavy to build, so they are
created lazily on the first call and cached for the life of the process. The
embedding function is injectable (see _build_collection) so tests can swap in a
tiny deterministic stub and run offline.
"""

import json
import os
import random
import time

# Default Sentence Transformer used both here and (as its ONNX twin
# "Xenova/all-MiniLM-L6-v2") in the browser mirror, so retrieval behaves the
# same in both. Small, fast, and good enough for short interrogation lines.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB is configured for cosine *distance* (1 - cosine similarity), so this
# is in [0, 2]. A query whose nearest neighbour is farther than this is treated
# as "no good match" and gets a generic fallback instead of a wrong answer.
# Tuned against data/suspect_qa.json with the real model: clear paraphrases land
# around 0.2-0.4 and off-topic input around 0.7+, so 0.6 keeps a comfortable
# margin on both sides (see the sweep used while authoring the dataset).
MATCH_DISTANCE_THRESHOLD = 0.6

# --- Simulated "thinking" latency for the static control ----------------------
# The static lookup answers almost instantly, which would let testers spot the
# pre-written condition by speed alone (and confounds the study). To match the
# dynamic LLM's feel, we sleep for a delay that scales with reply length (like
# real token generation) plus random jitter so it is never a constant tell.
# Starting values; calibrate against observed dynamic latencies in logs/.
SIM_LATENCY_BASE_S = 0.8       # fixed "thinking" floor before any text
SIM_LATENCY_PER_CHAR_S = 0.015 # per-character "generation" time
SIM_LATENCY_JITTER = 0.20      # +/- fraction applied to the total
SIM_LATENCY_MAX_S = 6.0        # hard ceiling so a long reply can't stall forever

# Path to the authored prompt -> response database.
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "suspect_qa.json")

# Used when no entry is close enough to the player's input. Cycled in order so
# repeated off-topic prompts don't echo the same line -- and since the cycle
# position only ever advances within a session, ordering the lines from polite
# to testy gives her patience a natural arc for free.
FALLBACK_LINES = [
    "I'm not sure what you want me to say to that.",
    "Could you be more specific, Inspector?",
    "I'm afraid I don't follow. Ask me plainly and I'll answer plainly.",
    "I don't see what that has to do with Charles, but do go on.",
    "You have me at a loss. Perhaps ask it another way.",
    "I've told you everything I can think of.",
    "We seem to be wandering, Inspector. Is there a question in there for me?",
    "If you have a question, Inspector, ask it; I am not a mind-reader.",
]

# Lazily built, then cached for the process lifetime.
_collection = None


def load_qa_data(path=_DATA_PATH):
    """
    Load and validate the list of {id, responses, repeat_responses, questions}
    entries from disk.

    Validation is deliberately fail-fast: a malformed database should crash at
    startup (warm_up), not answer players with a KeyError mid-study. Rules:
    every entry needs a unique id that does not start with "_" (underscore
    keys, like "_fallback", are reserved for counters in the visit_counts
    dict), a non-empty responses list and a non-empty questions list.
    repeat_responses is optional.
    """
    with open(path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    seen_ids = set()
    for entry in qa_data:
        entry_id = entry.get("id")
        if not entry_id or not isinstance(entry_id, str):
            raise ValueError(f"QA entry without a valid id: {entry!r}")
        if entry_id.startswith("_"):
            raise ValueError(
                f"QA entry id {entry_id!r} may not start with '_' (reserved "
                "for visit_counts bookkeeping keys)."
            )
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate QA entry id: {entry_id!r}")
        seen_ids.add(entry_id)
        if not entry.get("responses"):
            raise ValueError(f"QA entry {entry_id!r} has no responses.")
        if not entry.get("questions"):
            raise ValueError(f"QA entry {entry_id!r} has no example questions.")
    return qa_data


def _default_embedding_function():
    """
    The real embedding function: a Sentence Transformer wrapped for ChromaDB.

    Imported lazily so that importing this module (and the fast, stubbed tests)
    does not pull in sentence-transformers / torch unless retrieval is actually
    used with the real model.
    """
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )


def _build_collection(qa_data=None, embedding_fn=None):
    """
    Build an in-memory ChromaDB collection over the Q&A database and return it.

    Each example question becomes one document whose metadata carries its
    entry's id and response lists, so a nearest-neighbour query maps straight
    back to the lines to speak. ChromaDB metadata values must be scalars, so
    the lists are JSON-encoded strings; get_response decodes them. Keeping the
    responses inside the collection (rather than in a side table) keeps the
    test-injection contract simple: swapping in a stub collection swaps the
    whole database. `embedding_fn` is injectable: pass a stub in tests to
    avoid loading the real model; left as None it uses the Sentence Transformer.
    """
    import chromadb

    qa_data = qa_data if qa_data is not None else load_qa_data()
    embedding_fn = embedding_fn if embedding_fn is not None else _default_embedding_function()

    client = chromadb.Client()
    # A fresh, uniquely named collection so repeated builds (e.g. across tests)
    # never collide on the in-memory client.
    collection = client.create_collection(
        name=f"suspect_qa_{id(qa_data)}",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []
    for entry in qa_data:
        for i, question in enumerate(entry["questions"]):
            documents.append(question)
            metadatas.append({
                "entry_id": entry["id"],
                "responses": json.dumps(entry["responses"]),
                "repeat_responses": json.dumps(entry.get("repeat_responses", [])),
            })
            ids.append(f"{entry['id']}_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def _get_collection():
    """Return the cached collection, building it (with the real model) once."""
    global _collection
    if _collection is None:
        _collection = _build_collection()
    return _collection


def warm_up():
    """
    Build the embedding model and index ahead of time.

    Called at startup (see main.py) so the first study turn isn't slowed by the
    one-off model load and index build.
    """
    _get_collection()


def get_response(player_input, visit_counts=None):
    """
    Return the scripted NPC line whose stored question best matches the input.

    Parameters
    ----------
    player_input : str
        The raw text the player typed.
    visit_counts : dict, optional
        Per-session counters, keyed by entry id (ui.py passes its gr.State
        dict here and resets it with the session). The count for the matched
        entry decides which pre-written variant is spoken this time -- see
        _select_variant -- and "_fallback" tracks the generic-fallback cycle.
        Passing None keeps the call stateless: always responses[0].

    Returns
    -------
    str
        The chosen NPC reply.

    The match is the single nearest neighbour by cosine distance over the
    embedded example questions. If that distance is above
    MATCH_DISTANCE_THRESHOLD (or the input is empty), a generic fallback line is
    returned instead.
    """
    if not player_input or not player_input.strip():
        return _next_fallback(visit_counts)

    collection = _get_collection()
    result = collection.query(query_texts=[player_input], n_results=1)

    distances = result.get("distances") or [[]]
    metadatas = result.get("metadatas") or [[]]
    if not distances[0] or distances[0][0] > MATCH_DISTANCE_THRESHOLD:
        return _next_fallback(visit_counts)

    metadata = metadatas[0][0]
    return _select_variant(
        metadata["entry_id"],
        json.loads(metadata["responses"]),
        json.loads(metadata["repeat_responses"]),
        visit_counts,
    )


def _select_variant(entry_id, responses, repeat_responses, visit_counts):
    """
    Pick which pre-written line the matched entry speaks on this visit.

    Deterministic on purpose: two study participants who ask the same
    questions in the same order get the same transcript, and the stub-embedder
    tests can assert exact lines without seeding a RNG. The walk:

      visit 1..len(responses)   responses in order (responses[0] is the
                                canonical answer and, for the slip-carrier
                                entries, the only line containing the slip --
                                mirroring the dynamic condition, where each
                                nugget drops once)
      after that                repeat_responses cycle ("asked and answered"
                                lines), or the responses wrap around if an
                                entry has no repeat lines authored
    """
    if visit_counts is None:
        return responses[0]
    n = visit_counts.get(entry_id, 0)
    visit_counts[entry_id] = n + 1
    if n < len(responses):
        return responses[n]
    extra = n - len(responses)
    if repeat_responses:
        return repeat_responses[extra % len(repeat_responses)]
    return responses[extra % len(responses)]


def simulate_latency(reply):
    """
    Sleep for a human/LLM-plausible delay derived from the reply length, then
    return the actual time slept in milliseconds.

    Used only by the static control so its perceived response time matches the
    dynamic LLM condition. The delay is base + per-char * len(reply), scaled by
    a random jitter factor and capped at SIM_LATENCY_MAX_S.
    """
    target = SIM_LATENCY_BASE_S + SIM_LATENCY_PER_CHAR_S * len(reply or "")
    target *= random.uniform(1.0 - SIM_LATENCY_JITTER, 1.0 + SIM_LATENCY_JITTER)
    target = min(target, SIM_LATENCY_MAX_S)
    start = time.perf_counter()
    time.sleep(target)
    return (time.perf_counter() - start) * 1000.0


def _next_fallback(visit_counts):
    """Cycle through the generic fallbacks, tracking position if we're given a dict."""
    if visit_counts is None:
        return FALLBACK_LINES[0]
    count = visit_counts.get("_fallback", 0)
    visit_counts["_fallback"] = count + 1
    return FALLBACK_LINES[count % len(FALLBACK_LINES)]
