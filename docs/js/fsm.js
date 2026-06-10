// fsm.js
// =====
// Browser port of fsm.py. The Finite State Machine that governs the suspect's
// emotional state and rewrites the model's system prompt each turn.
//
// This mirrors the Python source one-to-one (fsm.py in the repo root is the
// canonical version). The only difference is the audience: here the rewritten
// system prompt is fed to the small in-browser model in llm.js instead of the
// local Wayfarer-12B server.

export const State = {
  CALM: "Calm",
  SUSPICIOUS: "Suspicious",
  DEFENSIVE: "Defensive",
};

// Ground truth for the scenario (used only if a verdict mechanic is added).
export const SUSPECT_IS_GUILTY = true;

// Pointed or probing questioning escalates Calm -> Suspicious.
const PROBING_TRIGGERS = [
  "alibi", "where were you", "lie", "lying", "accuse", "suspect",
  "evidence", "witness saw", "you were", "explain yourself", "story",
  "doesn't add up", "convenient",
];

// Direct accusation or pressure escalates Suspicious -> Defensive.
const ACCUSATORY_TRIGGERS = [
  "you killed", "you murdered", "murderer", "confess", "guilty",
  "arrest", "caught you", "liar", "prove", "you did it", "blood",
  "fingerprints", "weapon", "motive",
];

// Reassurance or backtracking de-escalates the suspect one step.
const DEESCALATION_TRIGGERS = [
  "sorry", "no offense", "no offence", "just asking", "calm down",
  "take your time", "i understand", "i believe you", "thank you",
  "appreciate", "off the record", "not accusing",
];

const BASE_PERSONA =
  "You are Eleanor Vance, a 41 year old gallery owner. Last night you hosted " +
  "a dinner party at your country house. One of your guests, Charles Whitmore, " +
  "was found dead in the study before dessert. You are being questioned by an " +
  "investigator. Stay fully in character. Never mention that you are an AI, " +
  "never break character, and keep replies to two or three sentences so the " +
  "interrogation stays brisk. Do not narrate actions in asterisks; speak only " +
  "as Eleanor would speak aloud.";

const STATE_OVERLAYS = {
  [State.CALM]:
    "Right now you feel completely safe and unsuspected. As far as you are " +
    "concerned you are simply a host helping the investigator understand a " +
    "terrible accident. You are warm, articulate and genuinely cooperative, " +
    "happy to recount the evening in detail and to volunteer small, harmless " +
    "facts. You have nothing on your mind to hide.",
  [State.SUSPICIOUS]:
    "Important and secret: you are in fact responsible for Charles's death, " +
    "and you must never admit this or state it outright. You now sense the " +
    "questions are aimed at you, and your instinct is to protect yourself. " +
    "You become guarded and measured, give shorter answers, deflect, and " +
    "gently point the investigator toward other guests who had reason to " +
    "dislike Charles. Reveal nothing that incriminates you.",
  [State.DEFENSIVE]:
    "Important and secret: you are in fact responsible for Charles's death, " +
    "and you are now cornered. You are agitated, indignant and emotional. You " +
    "may raise your voice, demand a lawyer, or accuse the investigator of " +
    "harassment. Under this pressure you occasionally let slip one oddly " +
    "specific detail about the study or the timeline that an innocent person " +
    "could not know, and you immediately try to cover it. You still never " +
    "openly confess.",
};

export class SuspectFSM {
  constructor() {
    this.state = State.CALM;
    this.history = [];
  }

  getState() {
    return this.state;
  }

  getSystemPrompt() {
    return `${BASE_PERSONA}\n\n${STATE_OVERLAYS[this.state]}`;
  }

  // Update the state based on the player's input and return the new state.
  // Escalation is checked before de-escalation, and the most severe match
  // wins, so a single hostile line moves the suspect toward Defensive rather
  // than being cancelled out by a polite word elsewhere in the sentence.
  transition(playerInput) {
    const text = playerInput.toLowerCase();
    const oldState = this.state;

    const accusatory = ACCUSATORY_TRIGGERS.some((t) => text.includes(t));
    const probing = PROBING_TRIGGERS.some((t) => text.includes(t));
    const deescalating = DEESCALATION_TRIGGERS.some((t) => text.includes(t));

    if (accusatory) {
      if (this.state === State.CALM) this.state = State.SUSPICIOUS;
      else if (this.state === State.SUSPICIOUS) this.state = State.DEFENSIVE;
      // Already Defensive: stays Defensive.
    } else if (probing) {
      if (this.state === State.CALM) this.state = State.SUSPICIOUS;
    } else if (deescalating) {
      if (this.state === State.DEFENSIVE) this.state = State.SUSPICIOUS;
      else if (this.state === State.SUSPICIOUS) this.state = State.CALM;
    }

    this.history.push({ input: playerInput, from: oldState, to: this.state });
    return this.state;
  }

  reset() {
    this.state = State.CALM;
    this.history = [];
  }
}
