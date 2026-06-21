// intent.js
// =====
// Browser port of intent_classifier.py. Turns one player turn into a structured,
// multi-axis Signal that the FSM (fsm.js) uses to pick the suspect's next state.
//
// Mirrors the Python module: the model scores each turn on several independent
// axes at once, and the FSM transitions only when the combination agrees. That
// is what keeps the two "cornered" reactions distinct — aggression alone never
// crosses the awareness boundary, only concrete evidence does.
//
// classifyIntent() asks the in-browser model (llm.js) for a strict JSON
// judgement; classifyKeywords() is a deterministic fallback with the same shape,
// used when the model is not ready or its JSON does not parse, so a turn is never
// lost. Note this is a SECOND model call per turn on top of the reply, which adds
// latency on the small in-browser model — acceptable for the demo.
//
// Asymmetry with the Python path: the Python classifier constrains llama-server's
// decoding with a json_schema (intent_classifier.RESPONSE_FORMAT) so even a
// roleplay-tuned model must emit a valid Signal. transformers.js has no grammar
// constraint, so here we rely on the small instruct model following the JSON
// prompt and lean on the keyword fallback when it does not.

import * as llm from "./llm.js";

const EVIDENCE_LEVELS = ["none", "weak", "strong"];
const ACCUSATION_LEVELS = ["none", "implied", "direct"];
const AGGRESSION_LEVELS = ["low", "medium", "high"];
const WARMTH_LEVELS = ["cold", "neutral", "warm"];

// Neutral defaults: a flat, unremarkable line. A partial/failed classification
// still yields a sane Signal rather than breaking the turn.
export function neutralSignal() {
  return {
    evidence: "none",
    accusation: "none",
    aggression: "low",
    warmth: "neutral",
    conscience: false,
    probing: false,
  };
}

function coerce(value, allowed, fallback) {
  return typeof value === "string" && allowed.includes(value.toLowerCase())
    ? value.toLowerCase()
    : fallback;
}

function signalFromObject(data) {
  return {
    evidence: coerce(data.evidence, EVIDENCE_LEVELS, "none"),
    accusation: coerce(data.accusation, ACCUSATION_LEVELS, "none"),
    aggression: coerce(data.aggression, AGGRESSION_LEVELS, "low"),
    warmth: coerce(data.warmth, WARMTH_LEVELS, "neutral"),
    conscience: Boolean(data.conscience),
    probing: Boolean(data.probing),
  };
}

// ---- Keyword fallback vocabularies (mirror intent_classifier.py) ------------
// Concrete proof or a caught contradiction — the only thing that crosses the
// awareness boundary, so kept deliberately specific.
const EVIDENCE_TERMS = [
  "fingerprint", "fingerprints", "your prints", "dna", "cctv", "footage",
  "camera", "the weapon", "the knife", "we found", "found the", "witness saw",
  "the maid saw", "you said earlier", "you just said", "you told me",
  "contradict", "doesn't add up", "story changed", "phone records",
];
const PROBING_TERMS = [
  "alibi", "where were you", "what time", "timeline", "explain yourself",
  "your story", "walk me through", "why were you", "how did you",
];
const DIRECT_ACCUSATION_TERMS = [
  "you killed", "you murdered", "you did it", "murderer", "you're guilty",
  "you are guilty", "confess", "i know you did", "caught you",
];
const IMPLIED_ACCUSATION_TERMS = [
  "suspect", "lying", "you lied", "liar", "hiding something", "not telling",
  "motive", "convenient",
];
const AGGRESSIVE_TERMS = [
  "shut up", "liar", "stop lying", "you disgust", "pathetic", "i'll make you",
  "you'll rot", "don't play", "answer me", "enough", "!!!",
];
const WARM_TERMS = [
  "take your time", "i understand", "i believe you", "no rush", "it's okay",
  "i'm sorry", "no offense", "no offence", "not accusing", "off the record",
  "calm down", "appreciate", "thank you",
];
const CONSCIENCE_TERMS = [
  "do the right thing", "tell me what happened", "tell the truth", "for them",
  "let it go", "you'll feel better", "his family", "her family", "get it off",
  "come clean", "make it right",
];

const hasAny = (text, terms) => terms.some((t) => text.includes(t));

export function classifyKeywords(playerInput) {
  const text = (playerInput || "").toLowerCase();

  let accusation = "none";
  if (hasAny(text, DIRECT_ACCUSATION_TERMS)) accusation = "direct";
  else if (hasAny(text, IMPLIED_ACCUSATION_TERMS)) accusation = "implied";

  return {
    evidence: hasAny(text, EVIDENCE_TERMS) ? "strong" : "none",
    accusation,
    aggression: hasAny(text, AGGRESSIVE_TERMS) ? "high" : "low",
    warmth: hasAny(text, WARM_TERMS) ? "warm" : "neutral",
    conscience: hasAny(text, CONSCIENCE_TERMS),
    probing: hasAny(text, PROBING_TERMS),
  };
}

// ---- Model-driven classification --------------------------------------------
const CLASSIFIER_SYSTEM_PROMPT =
  "You are a classifier for a detective interrogation game. Read the " +
  "investigator's latest line (with the suspect's previous line for context) " +
  "and rate it on six axes. Reply with ONLY a JSON object, no prose, using " +
  "exactly these keys and allowed values:\n" +
  '  "evidence": "none" | "weak" | "strong"   ' +
  "(strong = a concrete fact, physical proof, or catching the suspect in a " +
  "contradiction; weak = a vague or unsupported claim; none = no evidence)\n" +
  '  "accusation": "none" | "implied" | "direct"   ' +
  "(direct = openly says she did it; implied = hints she is guilty)\n" +
  '  "aggression": "low" | "medium" | "high"   ' +
  "(high = insults, threats, shouting)\n" +
  '  "warmth": "cold" | "neutral" | "warm"   ' +
  "(warm = reassuring, patient, empathetic)\n" +
  '  "conscience": true | false   ' +
  "(true = explicitly urges her to confess, come clean, or do the right thing)\n" +
  '  "probing": true | false   ' +
  "(true = a pointed investigative question about alibi, timeline, or motive)\n" +
  'Example: {"evidence":"strong","accusation":"direct","aggression":"low",' +
  '"warmth":"neutral","conscience":false,"probing":false}';

function extractJson(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) {
    throw new Error("no JSON object in classifier reply");
  }
  return JSON.parse(text.slice(start, end + 1));
}

// Classify the player's turn into a Signal. `history` is the prior
// [{ role, content }, ...]; the last assistant line is included so tone is
// judged in context. Falls back to classifyKeywords if the model is not ready
// or the reply is not usable JSON.
export async function classifyIntent(playerInput, history = []) {
  if (!llm.isReady()) return classifyKeywords(playerInput);

  let lastNpc = "";
  for (let i = history.length - 1; i >= 0; i -= 1) {
    if (history[i].role === "assistant") {
      lastNpc = history[i].content;
      break;
    }
  }

  const userBlock =
    (lastNpc ? `Suspect's previous line: ${lastNpc}\n` : "") +
    `Investigator's latest line: ${playerInput}`;

  try {
    const reply = await llm.generate(
      CLASSIFIER_SYSTEM_PROMPT,
      [{ role: "user", content: userBlock }],
      { maxNewTokens: 80, temperature: 0 },
    );
    return signalFromObject(extractJson(reply));
  } catch (err) {
    console.warn("intent classification fell back to keywords:", err);
    return classifyKeywords(playerInput);
  }
}
