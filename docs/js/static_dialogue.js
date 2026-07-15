// static_dialogue.js
// =================
// Browser port of static_dialogue.py: the model-free "Static Control" baseline,
// built as a semantic retrieval lookup. Nothing is generated — every reply is a
// fixed, pre-written line pulled verbatim from the Q&A database.
//
// How it works (mirrors the Python ChromaDB + Sentence Transformer version):
//   1. The Q&A database lives in static_qa_data.js. Each entry pairs an ordered
//      list of response variants (plus optional "asked and answered" repeat
//      lines) with example questions, embedded once with all-MiniLM-L6-v2 (the
//      same model the Python side uses, here as its ONNX twin
//      "Xenova/all-MiniLM-L6-v2" via transformers.js) and cached as unit vectors.
//   2. The player's input is embedded with the same model; we take the dot
//      product against every cached vector (they're L2-normalised, so the dot
//      product is the cosine similarity) and route the best hit into one of
//      three bands: a confident match speaks the entry's line; a near-miss --
//      slightly too far, or a coin-flip tie between two different entries --
//      gets an in-character clarifying line naming the closest topic; anything
//      farther gets a generic fallback. The margin check against the best
//      *different* entry is what stops a borderline question from routing to a
//      confident answer about the wrong topic.
//   3. On a confident match the entry's next unspoken variant is returned: the
//      first ask gets responses[0], later asks walk the remaining variants in
//      order, then the repeat_responses cycle. Position lives in the
//      per-session visitCounts, so the rotation is deterministic and resets
//      with the session. Clarify and fallback lines cycle the same way under
//      the reserved `_clarify` / `_fallback` counter keys.
//
// There is no ChromaDB in the browser; a linear scan over a few dozen vectors is
// instant. static_dialogue.py in the repo root is the canonical version.
// Unlike the Python app, the demo adds no artificial latency: the tabs here are
// openly labeled, so there is no blinded condition to disguise by timing.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
import { SUSPECT_QA } from "./static_qa_data.js";

// ONNX twin of the Python side's all-MiniLM-L6-v2, so retrieval matches.
const EMBED_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

// Routing bands, in cosine *similarity* space. The Python side works in cosine
// distance; similarity = 1 - distance, and a margin is a difference so it maps
// unchanged. Keep these in lock-step with static_dialogue.py:
//   MATCH_SIMILARITY_THRESHOLD   0.45 similarity == 0.55 distance
//   CLARIFY_SIMILARITY_THRESHOLD 0.30 similarity == 0.70 distance
//   MATCH_MARGIN                 0.08 in both spaces
const MATCH_SIMILARITY_THRESHOLD = 0.45;
const CLARIFY_SIMILARITY_THRESHOLD = 0.30;
const MATCH_MARGIN = 0.08;

// Spoken in the clarify band, with {topic} filled from the nearest entry's
// topic_hint. Cycled like FALLBACK_LINES so repeated ambiguity doesn't echo.
// Mirrors CLARIFY_TEMPLATES in static_dialogue.py.
const CLARIFY_TEMPLATES = [
  "If it's {topic} you're asking about, Inspector, say so plainly and I shall answer plainly.",
  "You'll forgive me -- is this about {topic}? Ask it straight and you'll have it straight.",
  "I can guess you mean {topic}, but I'd rather not answer a guess. Put the question properly.",
];

// Used when no entry is close enough to the player's input. Cycled in order so
// repeated off-topic prompts don't echo the same line — and since the cycle
// position only ever advances within a session, ordering the lines from polite
// to testy gives her patience a natural arc for free.
const FALLBACK_LINES = [
  "I'm not sure what you want me to say to that.",
  "Could you be more specific, Inspector?",
  "I'm afraid I don't follow. Ask me plainly and I'll answer plainly.",
  "I don't see what that has to do with Charles, but do go on.",
  "You have me at a loss. Perhaps ask it another way.",
  "I've told you everything I can think of.",
  "We seem to be wandering, Inspector. Is there a question in there for me?",
  "If you have a question, Inspector, ask it; I am not a mind-reader.",
];

let extractor = null;       // cached feature-extraction pipeline
let loadingPromise = null;  // guards against concurrent load() calls
let corpus = null;          // { vectors: number[][], entries: object[] }

// True once the embedding model and corpus vectors are ready.
export function isReady() {
  return corpus !== null;
}

// Load the embedding model and embed every database question once. Safe to call
// repeatedly; later calls reuse the in-flight/finished load. `onProgress`
// receives raw transformers.js progress events so a UI can show a bar.
export async function load(onProgress) {
  if (corpus) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = pipeline("feature-extraction", EMBED_MODEL_ID, {
    progress_callback: onProgress,
  })
    .then(async (pipe) => {
      extractor = pipe;

      const questions = [];
      const entries = [];
      for (const entry of SUSPECT_QA) {
        for (const question of entry.questions) {
          questions.push(question);
          entries.push(entry);
        }
      }
      // Mean-pool and L2-normalise so a dot product is the cosine similarity.
      const out = await extractor(questions, { pooling: "mean", normalize: true });
      corpus = { vectors: out.tolist(), entries };
    })
    .catch((err) => {
      loadingPromise = null; // allow a retry after a failed load
      throw err;
    });

  return loadingPromise;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// Return the scripted NPC line whose stored question best matches the input.
// `visitCounts` is the per-session counter dict app.js passes (and resets):
// the matched entry's count decides which pre-written variant is spoken this
// time (see selectVariant), and `_fallback` tracks the generic-fallback cycle.
// Async because the first call loads the embedding model (a small ~25 MB
// download, then cached).
export async function getResponse(playerInput, visitCounts) {
  if (!playerInput || !playerInput.trim()) return nextFallback(visitCounts);

  await load();

  const out = await extractor(playerInput, { pooling: "mean", normalize: true });
  const queryVec = out.tolist()[0];

  let bestScore = -Infinity;
  let bestEntry = null;
  for (let i = 0; i < corpus.vectors.length; i++) {
    const score = dot(queryVec, corpus.vectors[i]);
    if (score > bestScore) {
      bestScore = score;
      bestEntry = corpus.entries[i];
    }
  }

  if (bestEntry === null || bestScore < CLARIFY_SIMILARITY_THRESHOLD) {
    return nextFallback(visitCounts);
  }

  // Margin check, mirroring _route in static_dialogue.py: one entry usually
  // owns several stored questions, so other hits from the *same* entry are
  // corroboration; only the best *different* entry decides confidence.
  let bestOtherScore = -Infinity;
  for (let i = 0; i < corpus.vectors.length; i++) {
    if (corpus.entries[i].id === bestEntry.id) continue;
    const score = dot(queryVec, corpus.vectors[i]);
    if (score > bestOtherScore) bestOtherScore = score;
  }

  if (
    bestScore >= MATCH_SIMILARITY_THRESHOLD &&
    bestScore - bestOtherScore >= MATCH_MARGIN
  ) {
    return selectVariant(bestEntry, visitCounts);
  }
  return nextClarify(bestEntry, visitCounts);
}

// Pick which pre-written line the matched entry speaks on this visit. Mirrors
// _select_variant in static_dialogue.py exactly: visits walk `responses` in
// order (responses[0] is the canonical answer and the only line carrying a
// slip), then the `repeat_responses` cycle, wrapping back through `responses`
// if an entry has no repeat lines. Deterministic on purpose, so a session can
// be replayed.
function selectVariant(entry, visitCounts) {
  const responses = entry.responses;
  const repeats = entry.repeat_responses || [];
  if (!visitCounts) return responses[0];
  const n = visitCounts[entry.id] || 0;
  visitCounts[entry.id] = n + 1;
  if (n < responses.length) return responses[n];
  const extra = n - responses.length;
  if (repeats.length) return repeats[extra % repeats.length];
  return responses[extra % responses.length];
}

// Cycle the clarify templates, filling {topic} with the nearest entry's hint
// (or a default derived from its id). Mirrors _next_clarify in Python.
function nextClarify(entry, visitCounts) {
  const topic = entry.topic_hint || entry.id.replace(/_/g, " ");
  if (!visitCounts) return CLARIFY_TEMPLATES[0].replace("{topic}", topic);
  const count = visitCounts._clarify || 0;
  visitCounts._clarify = count + 1;
  return CLARIFY_TEMPLATES[count % CLARIFY_TEMPLATES.length].replace("{topic}", topic);
}

// Cycle through the generic fallbacks, tracking position if given a counts dict.
function nextFallback(visitCounts) {
  if (!visitCounts) return FALLBACK_LINES[0];
  const count = visitCounts._fallback || 0;
  visitCounts._fallback = count + 1;
  return FALLBACK_LINES[count % FALLBACK_LINES.length];
}
