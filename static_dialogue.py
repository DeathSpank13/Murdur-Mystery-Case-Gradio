"""
static_dialogue.py
==================
The model-free "Static Control" version of the interrogation, now built as a
semantic retrieval lookup. This is still the non-AI baseline that the dynamic
FSM plus LLM version is compared against in the user study: nothing is
generated, every reply is a fixed, pre-written line pulled verbatim from a
database.

How it works:

  1. A set of existing dialogue entries (prompt to response pairs) lives in
     data/suspect_qa.json. Each entry pairs one canonical NPC response with
     several natural-language example questions that should map to it. Those
     example questions are embedded once with a Sentence Transformer
     (all-MiniLM-L6-v2) and stored in a ChromaDB collection.
  2. When the player writes something, that text is embedded with the same
     model and ChromaDB returns the nearest neighbour by cosine distance.
  3. The response attached to that nearest neighbour is returned verbatim. If
     nothing is close enough (distance above MATCH_DISTANCE_THRESHOLD), a
     generic fallback line is returned instead.

This replaces the older best-match *keyword* tree. The win over keywords is
that paraphrases the script never anticipated ("remind me what you're called")
still route to the right answer by meaning rather than by literal word overlap.
It stays squarely the simple control, though: the suspect is factual or evasive
but never confesses. The guilt fact lives only in the dynamic FSM overlays (see
fsm.py), not here. The point of the comparison is to show what is gained, and
what is lost, when these fixed lines are replaced by a model whose persona is
steered by the FSM.

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

# ChromaDB is configured for cosine *distance* (1 - cosine similarity), so this
# is in [0, 2]. A query whose nearest neighbour is farther than this is treated
# as "no good match" and gets a generic fallback instead of a wrong answer.
# Tuned against data/suspect_qa.json with the real model: clear paraphrases land
# around 0.2-0.4 and off-topic input around 0.7+, so 0.6 keeps a comfortable
# margin on both sides (see the sweep used while authoring the dataset).
MATCH_DISTANCE_THRESHOLD = 0.6

# Path to the authored prompt -> response database.
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "suspect_qa.json")

# Used when no entry is close enough to the player's input. Cycled in order so
# repeated off-topic prompts don't echo the same line.
FALLBACK_LINES = [
    "I'm not sure what you want me to say to that.",
    "Could you be more specific, Inspector?",
    "I've told you everything I can think of.",
    "I'm afraid I don't follow. Ask me plainly and I'll answer plainly.",
]

# Lazily built, then cached for the process lifetime.
_collection = None


def load_qa_data(path=_DATA_PATH):
    """Load the list of {id, response, questions} entries from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    entry's canonical response, so a nearest-neighbour query maps straight back
    to the line to speak. `embedding_fn` is injectable: pass a stub in tests to
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
            metadatas.append({"response": entry["response"]})
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
        Accepted for interface compatibility with the previous keyword version
        (ui.py passes its per-session counts here) but unused: retrieval is
        stateless and returns the nearest neighbour's response verbatim. Only
        the generic-fallback cycle keeps a little state, under "_fallback".

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

    return metadatas[0][0]["response"]


def _next_fallback(visit_counts):
    """Cycle through the generic fallbacks, tracking position if we're given a dict."""
    if visit_counts is None:
        return FALLBACK_LINES[0]
    count = visit_counts.get("_fallback", 0)
    visit_counts["_fallback"] = count + 1
    return FALLBACK_LINES[count % len(FALLBACK_LINES)]
