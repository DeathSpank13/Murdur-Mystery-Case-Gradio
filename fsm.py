"""
fsm.py
======
Finite State Machine governing the suspect NPC's emotional state during the
interrogation.

Design premise
--------------
The player takes the role of an investigator. The NPC is a murder suspect
(Eleanor Vance) being questioned about the death of a guest at a dinner party
she hosted. Her demeanour shifts depending on how she is questioned, and the
active state decides which system prompt is sent to the language model. That
is the core idea of the project: the FSM rewrites the LLM's persona in real
time, so the "same" character behaves very differently from one exchange to
the next.

States
------
CALM        Cooperative and relaxed. Believes she is just a helpful witness.
SUSPICIOUS  Guarded and wary. Senses the questions are pointed at her.
DEFENSIVE   Cornered and agitated. Either hostile or pleading, and prone to
            small slips that a sharp investigator can exploit.
BREAKING    Her composure has finally collapsed under sustained pressure. This
            is the only state in which she actually confesses. It is reached
            only after she has been kept under accusation while already
            Defensive, so a confession has to be earned, not handed out.

Transitions are driven by keyword matching on the player's input. This is a
deliberately simple, inspectable trigger model: it is easy to explain, easy to
demonstrate live, and easy to compare against the static (non-AI) version.
"""

from enum import Enum


class State(Enum):
    """The three emotional states the suspect can occupy."""
    CALM = "Calm"
    SUSPICIOUS = "Suspicious"
    DEFENSIVE = "Defensive"
    BREAKING = "Breaking"


# Ground truth for the scenario, used only to score the player's final verdict
# in the user study. Eleanor is guilty. Note that the model is not told this in
# the Calm state (see STATE_OVERLAYS); the fact is gated behind the FSM so the
# suspect cannot leak what she does not yet "know" she is hiding.
SUSPECT_IS_GUILTY = True


# ---------------------------------------------------------------------------
# Transition trigger vocabularies
# ---------------------------------------------------------------------------
# Each list holds lowercase substrings. If any appears in the player's input,
# the corresponding transition fires. Keeping these as named constants makes
# the behaviour transparent and easy to tune during user testing (Phase 5).

# Pointed or probing questioning escalates Calm -> Suspicious.
PROBING_TRIGGERS = [
    "alibi", "where were you", "lie", "lying", "accuse", "suspect",
    "evidence", "witness saw", "you were", "explain yourself", "story",
    "doesn't add up", "convenient",
]

# Direct accusation or pressure escalates Suspicious -> Defensive.
ACCUSATORY_TRIGGERS = [
    "you killed", "you murdered", "murderer", "confess", "guilty",
    "arrest", "caught you", "liar", "prove", "you did it", "blood",
    "fingerprints", "weapon", "motive",
]

# Reassurance or backtracking de-escalates the suspect one step.
DEESCALATION_TRIGGERS = [
    "sorry", "no offense", "no offence", "just asking", "calm down",
    "take your time", "i understand", "i believe you", "thank you",
    "appreciate", "off the record", "not accusing",
]

# How many further accusatory turns she must endure while ALREADY Defensive
# before her composure breaks and she confesses. This is the main knob for how
# hard the confession is to reach: raise it to make her more stubborn, lower it
# (to 1) to make her crack the moment she is cornered. The accusatory turn that
# first pushes her into Defensive does not count, so the minimum to confess is
# Defensive plus this many sustained accusations.
DEFENSIVE_BREAK_THRESHOLD = 1


# ---------------------------------------------------------------------------
# Persona: a neutral base plus a per-state overlay
# ---------------------------------------------------------------------------
# Design choice: the guilt fact is deliberately NOT in the base persona.
#
# A local model held in a system prompt that says "you are the killer" on every
# turn tends to leak: it gets evasive when a genuinely innocent host would be
# relaxed, or it half confesses unprompted. That collapses the interrogation.
#
# Instead the base only fixes who Eleanor is and the facts of the scene, with no
# statement of guilt or innocence. Each state overlay then decides how much she
# knows and how much she is hiding. So the FSM is not just changing her tone, it
# is changing what she is concealing, which is the more interesting behaviour and
# the stronger story for the comparative study.
#
#   CALM        She behaves as an innocent, cooperative witness. The model is
#               not told she is guilty, so it cannot leak what it does not know.
#   SUSPICIOUS  The guilt knowledge is introduced here, framed as "you have
#               something to hide and you are concealing it." Her guardedness now
#               has a real reason behind it.
#   DEFENSIVE   Full pressure, and the one place where an incriminating slip is
#               allowed to surface for a sharp player to catch.
#   BREAKING    The dam breaks. This is the ONLY overlay that permits an open
#               confession; every other overlay forbids it. It is reached only
#               after sustained accusation while Defensive, so the confession is
#               the payoff for real pressure rather than a freebie.

BASE_PERSONA = (
    "You are Eleanor Vance, a 41 year old gallery owner. Last night you hosted "
    "a dinner party at your country house. One of your guests, Charles Whitmore, "
    "was found dead in the study before dessert. You are being questioned by an "
    "investigator. Stay fully in character. Never mention that you are an AI, "
    "never break character, and keep replies to two or three sentences so the "
    "interrogation stays brisk. Do not narrate actions in asterisks; speak only "
    "as Eleanor would speak aloud."
)

STATE_OVERLAYS = {
    State.CALM: (
        "Right now you feel completely safe and unsuspected. As far as you are "
        "concerned you are simply a host helping the investigator understand a "
        "terrible accident. You are warm, articulate and genuinely cooperative, "
        "happy to recount the evening in detail and to volunteer small, harmless "
        "facts. You have nothing on your mind to hide."
    ),
    State.SUSPICIOUS: (
        "Important and secret: you are in fact responsible for Charles's death, "
        "and you must never admit this or state it outright. You now sense the "
        "questions are aimed at you, and your instinct is to protect yourself. "
        "You become guarded and measured, give shorter answers, deflect, and "
        "gently point the investigator toward other guests who had reason to "
        "dislike Charles. Reveal nothing that incriminates you."
    ),
    State.DEFENSIVE: (
        "Important and secret: you are in fact responsible for Charles's death, "
        "and you are now cornered. You are agitated, indignant and emotional. You "
        "may raise your voice, demand a lawyer, or accuse the investigator of "
        "harassment. Under this pressure you occasionally let slip one oddly "
        "specific detail about the study or the timeline that an innocent person "
        "could not know, and you immediately try to cover it. You still never "
        "openly confess."
    ),
    State.BREAKING: (
        "Important: the pressure has finally become unbearable and your "
        "composure has collapsed. You can no longer keep up the act or hold the "
        "secret in. You break down and openly admit that you are responsible "
        "for Charles's death, and through the emotion you say plainly what you "
        "did and why. There is no more deflecting or denying now; this is a "
        "genuine confession."
    ),
}


class SuspectFSM:
    """
    Tracks and updates the suspect's emotional state.

    Typical use per player turn:
        previous = fsm.state
        new_state = fsm.transition(player_input)
        system_prompt = fsm.get_system_prompt()
    """

    def __init__(self):
        self.state = State.CALM
        # Counts consecutive accusatory turns endured while already Defensive.
        # When it reaches DEFENSIVE_BREAK_THRESHOLD she breaks and confesses.
        # Reset whenever she de-escalates, so backing off buys her composure.
        self.defensive_pressure = 0
        # A simple audit trail. Useful in Phase 5 when analysing how players
        # drove the conversation, and handy to show during a live demo.
        self.history = []

    def get_state(self):
        """Return the current State."""
        return self.state

    def get_system_prompt(self):
        """Return the full system prompt for the current state."""
        return f"{BASE_PERSONA}\n\n{STATE_OVERLAYS[self.state]}"

    def transition(self, player_input):
        """
        Update the state based on the player's input and return the new State.

        Escalation is checked before de-escalation, and the most severe match
        wins, so a single hostile line moves the suspect toward Defensive
        rather than being cancelled out by a polite word elsewhere in the
        sentence.
        """
        text = player_input.lower()
        old_state = self.state

        accusatory = any(trigger in text for trigger in ACCUSATORY_TRIGGERS)
        probing = any(trigger in text for trigger in PROBING_TRIGGERS)
        deescalating = any(trigger in text for trigger in DEESCALATION_TRIGGERS)

        if accusatory:
            # A direct accusation always pushes one step toward Defensive, and
            # sustained accusation while already cornered finally breaks her.
            if self.state == State.CALM:
                self.state = State.SUSPICIOUS
            elif self.state == State.SUSPICIOUS:
                self.state = State.DEFENSIVE
            elif self.state == State.DEFENSIVE:
                # Already cornered: count the sustained pressure. The turn that
                # first reached Defensive does not count, so she only confesses
                # after DEFENSIVE_BREAK_THRESHOLD further accusations.
                self.defensive_pressure += 1
                if self.defensive_pressure >= DEFENSIVE_BREAK_THRESHOLD:
                    self.state = State.BREAKING
            # Already Breaking: stays Breaking (the confession is out).
        elif probing:
            # Pointed questioning nudges Calm into Suspicious.
            if self.state == State.CALM:
                self.state = State.SUSPICIOUS
        elif deescalating:
            # Reassurance walks the suspect back one step and lets her recompose,
            # so the pressure toward a confession has to be rebuilt from there.
            self.defensive_pressure = 0
            if self.state == State.BREAKING:
                self.state = State.DEFENSIVE
            elif self.state == State.DEFENSIVE:
                self.state = State.SUSPICIOUS
            elif self.state == State.SUSPICIOUS:
                self.state = State.CALM

        self.history.append(
            {
                "input": player_input,
                "from": old_state.value,
                "to": self.state.value,
            }
        )
        return self.state

    def reset(self):
        """Return the suspect to the starting Calm state and clear history."""
        self.state = State.CALM
        self.defensive_pressure = 0
        self.history = []
