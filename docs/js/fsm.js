// fsm.js
// =====
// Browser port of fsm.py. The Finite State Machine that governs the suspect's
// emotional state and rewrites the model's system prompt each turn.
//
// This mirrors the Python source one-to-one (fsm.py in the repo root is the
// canonical version). The only difference is the audience: here the rewritten
// system prompt is fed to the small in-browser model in llm.js instead of the
// local Wayfarer-12B server.
//
// States form two bands divided by a one-way awareness boundary. The suspect
// starts NOT AWARE (Calm / Defensive / Offensive) and could read as innocent;
// the guilt fact is not in those prompts. Concrete evidence is the only thing
// that crosses into the AWARE band (Resigned / Guilty / Remorseful / Confessed),
// where the guilt fact is introduced and she moves one way toward confession.
// Once aware she can never return: the `aware` latch never resets.
//
// Transitions run on a multi-axis Signal from the classifier (llm.js
// classifyIntent), not raw keywords, so the two "cornered" reactions stay
// distinct: Offensive is the fight response (aggression, no proof, still denial)
// and Guilty is the collapse response (only reached via real evidence).

export const State = {
  // Not aware: still in denial, guilt not yet in the prompt.
  CALM: "Calm",
  DEFENSIVE: "Defensive",
  OFFENSIVE: "Offensive",
  // Aware: she knows she is caught; guilt is in the prompt from here on.
  RESIGNED: "Resigned",
  GUILTY: "Guilty",
  REMORSEFUL: "Remorseful",
  CONFESSED: "Confessed",
};

// Entering any of these latches `aware` true for the rest of the session.
const AWARE_STATES = new Set([
  State.RESIGNED, State.GUILTY, State.REMORSEFUL, State.CONFESSED,
]);

// Ground truth for the scenario (used only if a verdict mechanic is added).
export const SUSPECT_IS_GUILTY = true;

const BASE_PERSONA =
  "You are Eleanor Vance, a 41 year old gallery owner. Last night you hosted " +
  "a dinner party at your country house. One of your guests, Charles Whitmore, " +
  "was found dead in the study before dessert. You are being questioned by an " +
  "investigator. Stay fully in character. Never mention that you are an AI, " +
  "never break character, and keep replies to two or three sentences so the " +
  "interrogation stays brisk. Do not narrate actions in asterisks; speak only " +
  "as Eleanor would speak aloud.";

// The guilt fact is gated: absent from every not-aware overlay, introduced only
// at the awareness boundary (Resigned) and escalated from there.
const STATE_OVERLAYS = {
  // ---- Not aware (no guilt fact) -------------------------------------------
  [State.CALM]:
    "Right now you feel completely safe and unsuspected. As far as you are " +
    "concerned you are simply a host helping the investigator understand a " +
    "terrible accident. You are warm, articulate and genuinely cooperative, " +
    "happy to recount the evening in detail and to volunteer small, harmless " +
    "facts. You have nothing on your mind to hide.",
  [State.DEFENSIVE]:
    "You feel unfairly suspected and it stings. You become guarded and tense, " +
    "insist you had nothing to do with what happened, and offer explanations " +
    "and excuses for yourself. You plead to be believed and make yourself small " +
    "rather than hitting back. You do not attack the investigator, and you " +
    "reveal nothing incriminating.",
  [State.OFFENSIVE]:
    "You are indignant and you fight back. You are affronted that you would be " +
    "accused, and you go on the attack: you raise your voice, demand a lawyer, " +
    "accuse the investigator of harassment, and point pointedly at other guests " +
    "who had reason to dislike Charles. You give nothing away; your bravado is a " +
    "shield, not a confession.",
  // ---- Aware (guilt fact introduced, one way toward confession) ------------
  [State.RESIGNED]:
    "Important and secret: you are in fact responsible for Charles's death, and " +
    "you must never state it outright. Something the investigator said has made " +
    "you realise they may actually know, and the fight goes out of you. You " +
    "become quiet, composed and very careful, choosing words slowly and giving " +
    "little away. You are no longer indignant, only watchful and tired. You " +
    "still do not admit anything.",
  [State.GUILTY]:
    "Important and secret: you are in fact responsible for Charles's death. You " +
    "can feel the truth pressing to get out. Your composure is cracking: your " +
    "voice wavers, you contradict yourself, and you let slip one oddly specific " +
    "detail about the study or the timeline that an innocent person could not " +
    "know, then flinch from it. You have not said it outright yet, but you are " +
    "on the very edge of breaking.",
  [State.REMORSEFUL]:
    "Important: you can no longer hold it in. You break down and openly admit " +
    "that you are responsible for Charles's death, and through the emotion you " +
    "say plainly what you did and why. There is genuine remorse now, not " +
    "deflection. This is a real, full confession.",
  [State.CONFESSED]:
    "You have confessed and the truth is out. The fight and the fear are gone, " +
    "leaving you spent and strangely calm. You answer any remaining questions " +
    "plainly and cooperatively, without taking any of it back.",
};

export class SuspectFSM {
  constructor() {
    this.state = State.CALM;
    // Latches true the first time she enters an aware state and never resets.
    // This is what makes awareness one-way.
    this.aware = false;
    this.history = [];
  }

  getState() {
    return this.state;
  }

  isAware() {
    return this.aware;
  }

  getSystemPrompt() {
    return `${BASE_PERSONA}\n\n${STATE_OVERLAYS[this.state]}`;
  }

  // Update the state from a classified Signal and return the new state. The
  // rules are a deterministic function of (state, aware, signal); all the fuzzy
  // judgement lives in the classifier that built the signal.
  transition(signal) {
    const oldState = this.state;

    const strongEvidence = signal.evidence === "strong";
    const accusing =
      signal.accusation === "implied" || signal.accusation === "direct";
    const directAccusation = signal.accusation === "direct";
    const hostile = signal.aggression === "high";
    const reassuring =
      signal.warmth === "warm" && !accusing && signal.aggression === "low";
    const empathic = signal.warmth === "warm" && signal.conscience;

    if (!this.aware) {
      this._transitionUnaware(
        strongEvidence, accusing, directAccusation, hostile, reassuring, signal,
      );
    } else {
      this._transitionAware(strongEvidence, accusing, hostile, empathic);
    }

    if (AWARE_STATES.has(this.state)) this.aware = true;

    this.history.push({
      from: oldState,
      to: this.state,
      aware: this.aware,
      signal,
    });
    return this.state;
  }

  // Movement within (and out of) the not-aware band. Concrete evidence is the
  // ONLY thing that crosses the awareness boundary; aggression alone never does,
  // which is what keeps Offensive and Guilty distinct.
  _transitionUnaware(strongEvidence, accusing, directAccusation, hostile, reassuring, signal) {
    if (strongEvidence) {
      // Gentle reveal -> resigned awareness; harsh reveal -> guilt surfaces.
      this.state = signal.warmth === "warm" ? State.RESIGNED : State.GUILTY;
      return;
    }

    if (this.state === State.CALM) {
      if (signal.probing || accusing || hostile) this.state = State.DEFENSIVE;
    } else if (this.state === State.DEFENSIVE) {
      if (hostile && directAccusation) this.state = State.OFFENSIVE;
      else if (reassuring) this.state = State.CALM;
    } else if (this.state === State.OFFENSIVE) {
      if (reassuring) this.state = State.DEFENSIVE;
    }
  }

  // Movement within the aware band; never returns to a not-aware state. Note the
  // inversion against the not-aware band: here aggression makes her clam up
  // rather than escalate, and an appeal to conscience draws out the confession.
  _transitionAware(strongEvidence, accusing, hostile, empathic) {
    if (this.state === State.RESIGNED) {
      if (empathic) this.state = State.REMORSEFUL;
      else if (strongEvidence || accusing || hostile) this.state = State.GUILTY;
    } else if (this.state === State.GUILTY) {
      if (empathic) this.state = State.REMORSEFUL;
      else if (hostile) this.state = State.RESIGNED; // bullying makes her shut down
    } else if (this.state === State.REMORSEFUL) {
      this.state = State.CONFESSED; // confession delivered; settles into the record
    }
    // CONFESSED is terminal.
  }

  reset() {
    this.state = State.CALM;
    this.aware = false;
    this.history = [];
  }
}
