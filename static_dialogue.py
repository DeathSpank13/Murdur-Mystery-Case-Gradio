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
     model and ChromaDB returns the nearest neighbours by cosine distance.
     The result is routed into one of three bands (see _route): a confident
     match speaks the entry's line; a near-miss -- either slightly too far, or
     a coin-flip tie between two different entries -- gets an in-character
     clarifying line that names the closest topic; anything farther gets a
     generic fallback. The margin check between the best and the best
     *different* entry is what stops a borderline question from routing to a
     confident answer about the wrong topic.
  3. On a confident match the entry's next unspoken variant is returned: the
     first ask gets responses[0] (the canonical answer), later asks walk the
     remaining variants in order, and once those run out the repeat_responses
     cycle. The per-session visit_counts dict passed in by ui.py tracks the
     position, so the rotation is deterministic and resets with the session.
     The clarify and fallback lines cycle the same way, under the reserved
     "_clarify" / "_fallback" counter keys.

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

# Default Sentence Transformer used both here and (as its ONNX twin
# "Xenova/all-MiniLM-L6-v2") in the browser mirror, so retrieval behaves the
# same in both. Small, fast, and good enough for short interrogation lines.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval routing bands ---------------------------------------------------
# ChromaDB is configured for cosine *distance* (1 - cosine similarity), so all
# thresholds here are in [0, 2]. Tuned against data/suspect_qa.json with the
# real model: clear paraphrases land around 0.2-0.4 and off-topic input around
# 0.7+. The query result is routed into one of three bands:
#
#   match     d0 <= MATCH_DISTANCE_THRESHOLD and the best *different* entry is
#             at least MATCH_MARGIN farther -- speak the entry's line
#   clarify   d0 <= CLARIFY_DISTANCE_THRESHOLD, or the margin was too small --
#             ask an in-character clarifying question naming the nearest topic
#   fallback  anything farther -- generic "I don't follow" cycle
#
# (Mirror any change in docs/js/static_dialogue.js, which uses the same values
# in cosine *similarity* space: similarity = 1 - distance.)

# How many neighbours to fetch per query. Entries have ~6 example questions
# each, so the top of the list can be filled by one entry; 5 is enough to find
# the best *different* entry for the margin check.
QUERY_TOP_K = 5

# Nearest neighbour farther than this is never a confident match.
MATCH_DISTANCE_THRESHOLD = 0.55

# Upper edge of the near-miss band: beyond this the input is genuinely
# off-topic (observed 0.7+) and guessing a topic would itself feel weird.
CLARIFY_DISTANCE_THRESHOLD = 0.70

# Minimum distance gap between the best entry and the best *different* entry
# for a confident match. Same-topic paraphrases typically separate by >0.1
# with MiniLM; a smaller gap is a coin flip between two topics, and a wrong
# but confident answer is exactly the failure mode this exists to prevent.
MATCH_MARGIN = 0.08

# Spoken in the clarify band, with {topic} filled from the nearest entry's
# topic_hint. Cycled like FALLBACK_LINES so repeated ambiguity doesn't echo.
CLARIFY_TEMPLATES = [
    "If it's {topic} you're asking about, Inspector, say so plainly and I shall answer plainly.",
    "You'll forgive me -- is this about {topic}? Ask it straight and you'll have it straight.",
    "I can guess you mean {topic}, but I'd rather not answer a guess. Put the question properly.",
]

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
    keys, like "_fallback" and "_clarify", are reserved for counters in the
    visit_counts dict), a non-empty responses list and a non-empty questions
    list. repeat_responses is optional, as is topic_hint (a short in-character
    noun phrase spoken in the clarify band; when absent it is derived from the
    id).
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
        if "topic_hint" in entry and (
            not entry["topic_hint"] or not isinstance(entry["topic_hint"], str)
        ):
            raise ValueError(
                f"QA entry {entry_id!r} has an invalid topic_hint (must be a "
                "non-empty string when present)."
            )
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
                "topic_hint": entry.get("topic_hint") or entry["id"].replace("_", " "),
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


def _route(player_input):
    """
    Query the collection and decide which band the input lands in.

    Returns (band, metadata) where band is "match", "clarify" or "fallback",
    and metadata is the nearest entry's document metadata ("match" and
    "clarify") or None ("fallback"). All the routing thresholds live here so
    get_response and the test sweeps exercise one code path.

    The margin check: results are the nearest embedded example questions, and
    one entry usually owns several of them, so hits 2..k from the *same* entry
    as the best hit are corroboration, not ambiguity. Only the distance gap to
    the best *different* entry decides whether the match is confident.
    """
    collection = _get_collection()
    result = collection.query(
        query_texts=[player_input],
        # Clamp: tiny test databases can hold fewer documents than QUERY_TOP_K,
        # and some ChromaDB versions raise when n_results exceeds the count.
        n_results=min(QUERY_TOP_K, collection.count()),
    )

    distances = (result.get("distances") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    if not distances:
        return "fallback", None

    best_distance = distances[0]
    best_metadata = metadatas[0]
    if best_distance > CLARIFY_DISTANCE_THRESHOLD:
        return "fallback", None

    # Distance to the best entry that is NOT the matched one (None if every
    # returned neighbour belongs to the same entry -- maximal confidence).
    other_distance = next(
        (d for d, m in zip(distances, metadatas)
         if m["entry_id"] != best_metadata["entry_id"]),
        None,
    )
    if best_distance <= MATCH_DISTANCE_THRESHOLD and (
        other_distance is None or other_distance - best_distance >= MATCH_MARGIN
    ):
        return "match", best_metadata
    return "clarify", best_metadata


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
        _select_variant -- and the reserved "_fallback" / "_clarify" keys
        track the fallback and clarify cycles. Passing None keeps the call
        stateless: always responses[0] / the first template.

    Returns
    -------
    str
        The chosen NPC reply.

    The input is embedded and routed by _route: a confident nearest-neighbour
    match speaks the entry's next variant; a near-miss or a too-close tie
    between two entries gets a clarifying line naming the nearest topic; a
    clear miss (or empty input) gets a generic fallback line.
    """
    if not player_input or not player_input.strip():
        return _next_fallback(visit_counts)

    band, metadata = _route(player_input)
    if band == "fallback":
        return _next_fallback(visit_counts)
    if band == "clarify":
        return _next_clarify(metadata["topic_hint"], visit_counts)
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


def _next_clarify(topic_hint, visit_counts):
    """Cycle the clarify templates, filling {topic} with the nearest entry's hint."""
    if visit_counts is None:
        return CLARIFY_TEMPLATES[0].format(topic=topic_hint)
    count = visit_counts.get("_clarify", 0)
    visit_counts["_clarify"] = count + 1
    return CLARIFY_TEMPLATES[count % len(CLARIFY_TEMPLATES)].format(topic=topic_hint)


def _next_fallback(visit_counts):
    """Cycle through the generic fallbacks, tracking position if we're given a dict."""
    if visit_counts is None:
        return FALLBACK_LINES[0]
    count = visit_counts.get("_fallback", 0)
    visit_counts["_fallback"] = count + 1
    return FALLBACK_LINES[count % len(FALLBACK_LINES)]
