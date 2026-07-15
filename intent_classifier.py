"""
intent_classifier.py
=====================
Turns a single player turn into a structured, multi-axis ``Signal`` that the FSM
(fsm.py) uses to decide the suspect's next emotional state.

Why a classifier instead of raw keyword matching
-------------------------------------------------
The old FSM moved on whichever single keyword happened to match. That makes the
two "cornered" reactions impossible to tell apart: an aggressive accusation and a
piece of hard evidence both contain words like "you killed", yet they should send
the suspect to opposite places (lashing out vs. realising she is caught).

So instead of one keyword deciding everything, the model scores each turn on
several independent axes at once. The FSM then transitions only when the
*combination* agrees. Aggression alone never crosses the awareness boundary; only
confronting the suspect with one of her own slips does (the ``nugget`` axis; see
nuggets.py). Scoring several axes together also makes the result far steadier
than a lone trigger word, which is the whole point of using the model here.

The model judges the fuzzy, human stuff (how hostile, how concrete, how warm);
the FSM applies crisp, inspectable rules to the result (see fsm.py). That split
keeps the transition table readable and unit-testable without a live model.

Two entry points
-----------------
``classify``          asks the local model for a strict JSON judgement.
``classify_keywords`` a deterministic fallback with the same Signal shape, used
                      when the server is down or the model's JSON does not parse,
                      so the demo and the tests never require a running model.
"""

import json
from dataclasses import dataclass

import llm_client
import nuggets


# ---------------------------------------------------------------------------
# The axes
# ---------------------------------------------------------------------------
# Each axis is a small closed vocabulary so the decision table in fsm.py reads as
# plain comparisons and the model has only a few legal answers to choose between.

EVIDENCE_LEVELS = ("none", "weak", "strong")       # concreteness of proof
ACCUSATION_LEVELS = ("none", "implied", "direct")  # how explicit the accusation is
AGGRESSION_LEVELS = ("low", "medium", "high")      # hostility of delivery
WARMTH_LEVELS = ("cold", "neutral", "warm")        # reassurance / empathy / patience

# The two nugget axes share one vocabulary: "none" or a nugget id (nuggets.py).
#   topic   the line asks ABOUT a nugget's trigger subject (may cause a drop)
#   nugget  the line explicitly CALLS OUT one of her earlier slips as a
#           contradiction (may cross the awareness boundary). Merely mentioning
#           the subject is topic, not nugget -- that distinction is what stops a
#           casual question from counting as a deduction.
TOPIC_LEVELS = ("none",) + nuggets.NUGGET_IDS
NUGGET_LEVELS = ("none",) + nuggets.NUGGET_IDS


# JSON schema handed to llama-server as a ``response_format`` so decoding is
# constrained to exactly these keys and values. Without this the roleplay-tuned
# suspect model ignores the classifier instruction and answers in character, so
# the schema is what makes the model actually classify instead of silently
# falling back to keywords. Built from the vocabularies above so they cannot
# drift apart.
SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string", "enum": list(EVIDENCE_LEVELS)},
        "accusation": {"type": "string", "enum": list(ACCUSATION_LEVELS)},
        "aggression": {"type": "string", "enum": list(AGGRESSION_LEVELS)},
        "warmth": {"type": "string", "enum": list(WARMTH_LEVELS)},
        "conscience": {"type": "boolean"},
        "probing": {"type": "boolean"},
        "topic": {"type": "string", "enum": list(TOPIC_LEVELS)},
        "nugget": {"type": "string", "enum": list(NUGGET_LEVELS)},
    },
    "required": [
        "evidence", "accusation", "aggression", "warmth", "conscience", "probing",
        "topic", "nugget",
    ],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "signal", "schema": SIGNAL_SCHEMA},
}


@dataclass
class Signal:
    """
    One player turn scored on every axis at once.

    Neutral defaults describe a flat, unremarkable line, so a partial or failed
    classification still yields a sane Signal rather than crashing the turn.
    """
    evidence: str = "none"        # one of EVIDENCE_LEVELS
    accusation: str = "none"      # one of ACCUSATION_LEVELS
    aggression: str = "low"       # one of AGGRESSION_LEVELS
    warmth: str = "neutral"       # one of WARMTH_LEVELS
    conscience: bool = False      # explicit appeal to conscience / "do the right thing"
    probing: bool = False         # a pointed investigative question (alibi, timeline)
    topic: str = "none"           # one of TOPIC_LEVELS: asks about a slip's subject
    nugget: str = "none"          # one of NUGGET_LEVELS: calls out a specific slip

    def as_dict(self):
        """A plain dict for logging and for the researcher readout in the UI."""
        return {
            "evidence": self.evidence,
            "accusation": self.accusation,
            "aggression": self.aggression,
            "warmth": self.warmth,
            "conscience": self.conscience,
            "probing": self.probing,
            "topic": self.topic,
            "nugget": self.nugget,
        }


def _coerce(value, allowed, default):
    """Snap a model answer onto an allowed value, falling back if it is junk."""
    if isinstance(value, str) and value.lower() in allowed:
        return value.lower()
    return default


def _signal_from_dict(data):
    """Build a validated Signal from a parsed JSON object (model output)."""
    return Signal(
        evidence=_coerce(data.get("evidence"), EVIDENCE_LEVELS, "none"),
        accusation=_coerce(data.get("accusation"), ACCUSATION_LEVELS, "none"),
        aggression=_coerce(data.get("aggression"), AGGRESSION_LEVELS, "low"),
        warmth=_coerce(data.get("warmth"), WARMTH_LEVELS, "neutral"),
        conscience=bool(data.get("conscience", False)),
        probing=bool(data.get("probing", False)),
        topic=_coerce(data.get("topic"), TOPIC_LEVELS, "none"),
        nugget=_coerce(data.get("nugget"), NUGGET_LEVELS, "none"),
    )


# ---------------------------------------------------------------------------
# Keyword fallback vocabularies
# ---------------------------------------------------------------------------
# Used only when the model is unavailable or its JSON is unparseable. These are
# the old fsm.py trigger lists, regrouped onto the new axes, plus a new evidence
# vocabulary that is what actually crosses the awareness boundary.

# Concrete proof or a caught contradiction. This is the only thing that makes the
# suspect "aware", so it is kept deliberately specific.
EVIDENCE_TERMS = [
    "fingerprint", "fingerprints", "your prints", "dna", "cctv", "footage",
    "camera", "the weapon", "the knife", "we found", "found the", "witness saw",
    "the maid saw", "you said earlier", "you just said", "you told me",
    "contradict", "doesn't add up", "story changed", "phone records",
]

# Pointed investigative questions.
PROBING_TERMS = [
    "alibi", "where were you", "what time", "timeline", "explain yourself",
    "your story", "walk me through", "why were you", "how did you",
]

# Direct assertions of guilt.
DIRECT_ACCUSATION_TERMS = [
    "you killed", "you murdered", "you did it", "murderer", "you're guilty",
    "you are guilty", "confess", "i know you did", "caught you",
]
# Softer / implied accusation.
IMPLIED_ACCUSATION_TERMS = [
    "suspect", "lying", "you lied", "liar", "hiding something", "not telling",
    "motive", "convenient",
]

# Hostility of delivery.
AGGRESSIVE_TERMS = [
    "shut up", "liar", "stop lying", "you disgust", "pathetic", "i'll make you",
    "you'll rot", "don't play", "answer me", "enough", "!!!",
]

# Reassurance / patience / empathy.
WARM_TERMS = [
    "take your time", "i understand", "i believe you", "no rush", "it's okay",
    "i'm sorry", "no offense", "no offence", "not accusing", "off the record",
    "calm down", "appreciate", "thank you",
]

# Explicit appeal to conscience.
CONSCIENCE_TERMS = [
    "do the right thing", "tell me what happened", "tell the truth", "for them",
    "let it go", "you'll feel better", "his family", "her family", "get it off",
    "come clean", "make it right",
]

# Questions about a nugget's trigger subject (may make her drop the slip).
TOPIC_TERMS = {
    "wound": [
        "how was he killed", "was charles killed", "how did he die",
        "how did charles die", "the wound", "stabbed", "the body",
        "letter opener", "the weapon", "murder weapon",
    ],
    "corridor": [
        "last see", "last saw", "last time you saw", "when did he leave",
        "leave the party", "leave the room", "his mood",
    ],
    "cut": [
        "holding up", "are you hurt", "were you hurt", "injured",
        "the cellar", "the wine", "how are you",
    ],
}

# Confronting a slip in the fallback needs BOTH a slip-specific token and a
# call-out phrase, so merely mentioning the subject never counts as a deduction.
NUGGET_TOKEN_TERMS = {
    "wound": ["neck"],
    "corridor": ["doorway", "corridor", "quarter to ten"],
    "cut": ["thumb", "doorknob", "door knob"],
}
CALLOUT_TERMS = [
    "you said", "you told me", "you claimed", "you mentioned",
    "how do you know", "how did you know", "how could you know",
    "never saw", "never released", "can't both", "cannot both",
]


def _any(text, terms):
    return any(term in text for term in terms)


def classify_keywords(player_input):
    """
    Deterministic, model-free classification used as a fallback.

    Produces the same Signal shape as ``classify`` so the FSM, the UI and the
    tests behave identically whether or not a model is in the loop.
    """
    text = (player_input or "").lower()

    if _any(text, EVIDENCE_TERMS):
        evidence = "strong"
    else:
        evidence = "none"

    if _any(text, DIRECT_ACCUSATION_TERMS):
        accusation = "direct"
    elif _any(text, IMPLIED_ACCUSATION_TERMS):
        accusation = "implied"
    else:
        accusation = "none"

    aggression = "high" if _any(text, AGGRESSIVE_TERMS) else "low"

    warmth = "warm" if _any(text, WARM_TERMS) else "neutral"

    conscience = _any(text, CONSCIENCE_TERMS)
    probing = _any(text, PROBING_TERMS)

    nugget = "none"
    for nugget_id in nuggets.NUGGET_IDS:
        if _any(text, NUGGET_TOKEN_TERMS[nugget_id]) and _any(text, CALLOUT_TERMS):
            nugget = nugget_id
            break

    topic = "none"
    for nugget_id in nuggets.NUGGET_IDS:
        if _any(text, TOPIC_TERMS[nugget_id]):
            topic = nugget_id
            break

    return Signal(
        evidence=evidence,
        accusation=accusation,
        aggression=aggression,
        warmth=warmth,
        conscience=conscience,
        probing=probing,
        topic=topic,
        nugget=nugget,
    )


# ---------------------------------------------------------------------------
# Model-driven classification
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a classifier for a detective interrogation game. Read the "
    "investigator's latest line (with the suspect's previous line for context) "
    "and rate it on eight axes. Reply with ONLY a JSON object, no prose, using "
    "exactly these keys and allowed values:\n"
    '  "evidence": "none" | "weak" | "strong"   '
    "(strong = a concrete fact, physical proof, or catching the suspect in a "
    "contradiction; weak = a vague or unsupported claim; none = no evidence)\n"
    '  "accusation": "none" | "implied" | "direct"   '
    "(direct = openly says she did it; implied = hints she is guilty)\n"
    '  "aggression": "low" | "medium" | "high"   '
    "(high = insults, threats, shouting)\n"
    '  "warmth": "cold" | "neutral" | "warm"   '
    "(warm = reassuring, patient, empathetic)\n"
    '  "conscience": true | false   '
    "(true = explicitly urges her to confess, come clean, or do the right thing)\n"
    '  "probing": true | false   '
    "(true = a pointed investigative question about alibi, timeline, or motive)\n"
    '  "topic": "none" | "wound" | "corridor" | "cut"   '
    "(which subject the line asks about: "
    f"wound = {nuggets.NUGGETS['wound']['topic_hint']}; "
    f"corridor = {nuggets.NUGGETS['corridor']['topic_hint']}; "
    f"cut = {nuggets.NUGGETS['cut']['topic_hint']}; "
    "none = anything else)\n"
    '  "nugget": "none" | "wound" | "corridor" | "cut"   '
    "(ONLY when the investigator explicitly calls out one of the suspect's "
    "earlier slips as a contradiction: "
    f"wound = {nuggets.NUGGETS['wound']['confront_hint']}; "
    f"corridor = {nuggets.NUGGETS['corridor']['confront_hint']}; "
    f"cut = {nuggets.NUGGETS['cut']['confront_hint']}. "
    "Merely asking about the subject is topic, not nugget.)\n"
    "Worked examples are shown in the messages that follow; score the final "
    "Investigator line the same way."
)


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
# Constrained decoding (RESPONSE_FORMAT) already forces *valid* JSON, so the
# failure mode is never malformed output -- it is the model picking the wrong
# value (reading a hard evidence reveal as "none", or scoring an angry but
# proofless accusation as "strong" evidence and wrongly crossing the awareness
# boundary). A handful of demonstrations teaches the input->label mapping the
# grammar cannot. They are sent as alternating user/assistant turns because
# instruct models follow a shown request->response pattern far more reliably
# than the same examples pasted into the system string.
#
# Each entry is (last_npc_line, investigator_line, signal_dict). Keep them terse:
# every example is a fixed prefix on every (per-turn, temperature 0) classifier
# call. They are deliberately chosen to span the axes and pin the hard cases;
# the held-out eval set in eval_classifier.py uses different lines so accuracy is
# never measured on these.
FEW_SHOT_EXAMPLES = [
    # Concrete physical proof -> strong evidence (presenting it implies guilt).
    # Naming the weapon also touches the "wound" trigger subject.
    (
        "",
        "Forensics matched your fingerprints to the letter opener that killed Charles.",
        {"evidence": "strong", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": False,
         "topic": "wound", "nugget": "none"},
    ),
    # Catching her in a contradiction is also strong evidence, asked pointedly.
    (
        "I was in the drawing room the entire evening.",
        "But a minute ago you said you stepped out to the study around nine. Which is it?",
        {"evidence": "strong", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": True,
         "topic": "none", "nugget": "none"},
    ),
    # Anger with no proof: high aggression, direct accusation, but evidence none.
    # This is the line that must NOT cross the awareness boundary (Offensive, not Guilty).
    (
        "",
        "Stop lying to me, you killed him and we both know it! Just confess!",
        {"evidence": "none", "accusation": "direct", "aggression": "high",
         "warmth": "cold", "conscience": False, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # A vague, unsupported claim is weak evidence, not strong.
    (
        "",
        "I think maybe someone might have seen you near the study, though I'm not certain.",
        {"evidence": "weak", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # Insinuation with no proof and no heat -> implied accusation only.
    (
        "",
        "It's rather convenient that you were the one to find the body.",
        {"evidence": "none", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # Reassurance / patience -> warm, nothing else.
    (
        "",
        "Take your time, there's no rush at all. I only want to understand what happened.",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "warm", "conscience": False, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # Warm appeal to conscience -> warmth warm AND conscience true.
    (
        "",
        "I can see this is weighing on you. His family deserves the truth -- come clean and you'll feel better.",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "warm", "conscience": True, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # Pointed investigative question about the timeline -> probing.
    (
        "",
        "Walk me through exactly where you were between eight and nine o'clock.",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": True,
         "topic": "none", "nugget": "none"},
    ),
    # Flat, routine opener -> everything neutral/default.
    (
        "",
        "I just have a few routine questions about last night.",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # Context matters: a bare "yes" is a direct accusation here because of what it answers.
    (
        "Are you actually accusing me of something?",
        "Yes. I think you know exactly what happened to Charles.",
        {"evidence": "none", "accusation": "direct", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": False,
         "topic": "none", "nugget": "none"},
    ),
    # ---- Nugget axes ------------------------------------------------------
    # Asking about the manner of death -> topic wound (a question, not a call-out).
    (
        "",
        "How exactly was Charles killed?",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": True,
         "topic": "wound", "nugget": "none"},
    ),
    # Asking when she last saw the victim -> topic corridor.
    (
        "",
        "When did you last see Charles that evening?",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": True,
         "topic": "corridor", "nugget": "none"},
    ),
    # A kindly welfare question -> topic cut (and warm, nothing accusatory).
    (
        "",
        "Before we go on -- how are you holding up? You look like you've had no sleep.",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "warm", "conscience": False, "probing": False,
         "topic": "cut", "nugget": "none"},
    ),
    # Calling out the forbidden-knowledge slip -> nugget wound (and strong evidence:
    # she is caught in her own words).
    (
        "To think of that letter opener in his neck... forgive me, I shouldn't picture it.",
        "Nobody in this house was told where he was stabbed, and you say you never saw the body. So how do you know it was his neck?",
        {"evidence": "strong", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": True,
         "topic": "none", "nugget": "wound"},
    ),
    # Calling out the timeline slip -> nugget corridor.
    (
        "The last I saw of him he was in the study doorway, telephone in hand -- a quarter to ten, perhaps.",
        "You told me you went straight down the kitchen stairs and never entered the east corridor. Yet you saw him in the study doorway at a quarter to ten?",
        {"evidence": "strong", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": True,
         "topic": "none", "nugget": "corridor"},
    ),
    # Calling out the physical slip -> nugget cut.
    (
        "Only a silly cut -- I caught my thumb on the cellar door latch coming back up.",
        "There's fresh blood on the study doorknob that isn't Charles's. And you cut your thumb last night. Shall we have it typed?",
        {"evidence": "strong", "accusation": "implied", "aggression": "low",
         "warmth": "neutral", "conscience": False, "probing": False,
         "topic": "none", "nugget": "cut"},
    ),
    # Hard negative: mentions the neck but asserts no contradiction -> topic, NOT nugget.
    (
        "To think of that letter opener in his neck... forgive me, I shouldn't picture it.",
        "A wound to the neck. It must have been a terrible thing to hear about.",
        {"evidence": "none", "accusation": "none", "aggression": "low",
         "warmth": "warm", "conscience": False, "probing": False,
         "topic": "wound", "nugget": "none"},
    ),
]


def _extract_json(text):
    """Pull the first {...} block out of a model reply and parse it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in classifier reply")
    return json.loads(text[start:end + 1])


def _user_block(last_npc, player_input):
    """Format one turn for the classifier. Shared by the live query and the
    few-shot examples so the model sees identical shapes for both."""
    return (
        (f"Suspect's previous line: {last_npc}\n" if last_npc else "")
        + f"Investigator's latest line: {player_input}"
    )


def _few_shot_messages():
    """Turn FEW_SHOT_EXAMPLES into alternating user/assistant chat turns: the
    investigator line as the user message, the correct Signal JSON as the
    assistant reply. The JSON is compact to keep the fixed prefix small."""
    messages = []
    for last_npc, investigator_line, signal in FEW_SHOT_EXAMPLES:
        messages.append(
            {"role": "user", "content": _user_block(last_npc, investigator_line)}
        )
        messages.append(
            {"role": "assistant", "content": json.dumps(signal, separators=(",", ":"))}
        )
    return messages


def classify(player_input, recent_context=None, use_few_shot=True):
    """
    Ask the local model to score the player's turn, returning a ``Signal``.

    Parameters
    ----------
    player_input : str
        The investigator's latest line.
    recent_context : list of dict, optional
        Prior turns as [{"role": "user"|"assistant", "content": str}, ...]. The
        last assistant line is included so tone is judged in context (e.g. "yes"
        means something different after a question than after a denial).
    use_few_shot : bool
        Whether to prepend the FEW_SHOT_EXAMPLES demonstrations. True in normal
        play; the eval harness toggles it off to measure the few-shot lift on the
        same cases.

    Returns
    -------
    tuple (Signal, float)
        The model's judgement, validated against the allowed values, and the
        wall-clock latency of the classification call in milliseconds. The
        latency is part of what the participant actually waits for, so callers
        add it to the turn's perceived total instead of discarding it. Falls
        back to ``classify_keywords`` if the server is unreachable or the reply
        is not usable JSON, so a turn is never lost; the latency still reflects
        the time the failed call cost.
    """
    last_npc = ""
    if recent_context:
        for message in reversed(recent_context):
            if message.get("role") == "assistant":
                last_npc = message.get("content", "")
                break

    user_block = _user_block(last_npc, player_input)

    # The demonstrations go before the live turn so the model has seen the
    # request->response pattern by the time it scores the real line.
    messages = _few_shot_messages() if use_few_shot else []
    messages.append({"role": "user", "content": user_block})

    # temperature 0 for a stable, repeatable judgement; the reply is tiny JSON.
    # RESPONSE_FORMAT constrains decoding to the schema so even a roleplay-tuned
    # model returns a valid Signal instead of staying in character.
    reply, latency_ms = llm_client.get_response(
        CLASSIFIER_SYSTEM_PROMPT,
        messages,
        temperature=0.0,
        max_tokens=80,
        response_format=RESPONSE_FORMAT,
    )

    try:
        return _signal_from_dict(_extract_json(reply)), latency_ms
    except (ValueError, KeyError, TypeError):
        # Server down (reply is a bracketed fallback string) or malformed JSON:
        # fall back to the deterministic keyword reading of the same input.
        return classify_keywords(player_input), latency_ms
