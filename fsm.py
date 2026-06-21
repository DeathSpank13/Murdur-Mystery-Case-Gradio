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

The awareness boundary
----------------------
The states form two bands separated by a one-way door:

  NOT AWARE   CALM, DEFENSIVE, OFFENSIVE
              She is reacting to being questioned but is still in denial, and
              could read as innocent. The model is NOT told she is guilty, so it
              cannot leak what it does not "know".

  AWARE       RESIGNED, GUILTY, REMORSEFUL, CONFESSED
              She has realised she is caught. The guilt fact enters the system
              prompt here and she moves, one way, toward confession.

Once she becomes aware she can never return to a not-aware state: the ``aware``
latch flips True and never resets. This is the irreversibility the design calls
for; backing off inside the aware band can calm her, but cannot un-know what she
now knows.

The two "cornered" reactions are deliberately distinct:
  OFFENSIVE  fight  -- reached by raw aggression with no new proof; still denial.
  GUILTY     collapse -- reached only by concrete evidence, which is the single
             thing that crosses the awareness boundary.

Transitions are driven by a multi-axis ``Signal`` (intent_classifier.py), not by
raw keyword matching. The model scores each turn on several axes; this FSM
applies the crisp, inspectable rules below to the *combination* of those axes.
Keeping the rules here as plain comparisons means the behaviour stays easy to
explain, demonstrate live, and unit-test without a running model.
"""

from enum import Enum


class State(Enum):
    """The seven emotional states the suspect can occupy."""
    # Not aware: still in denial, guilt not yet in the prompt.
    CALM = "Calm"
    DEFENSIVE = "Defensive"
    OFFENSIVE = "Offensive"
    # Aware: she knows she is caught; guilt is in the prompt from here on.
    RESIGNED = "Resigned"
    GUILTY = "Guilty"
    REMORSEFUL = "Remorseful"
    CONFESSED = "Confessed"


# States that sit on the far side of the awareness boundary. Entering any of
# these latches ``aware`` True for the rest of the session.
AWARE_STATES = frozenset(
    {State.RESIGNED, State.GUILTY, State.REMORSEFUL, State.CONFESSED}
)


# Ground truth for the scenario, used only to score the player's final verdict
# in the user study. Eleanor is guilty. Note that the model is not told this in
# the not-aware states (see STATE_OVERLAYS); the fact is gated behind the FSM so
# the suspect cannot leak what she does not yet "know" she is hiding.
SUSPECT_IS_GUILTY = True


# ---------------------------------------------------------------------------
# Persona: a neutral base plus a per-state overlay
# ---------------------------------------------------------------------------
# Design choice: the guilt fact is deliberately NOT in the base persona, nor in
# any not-aware overlay.
#
# A local model told "you are the killer" on every turn tends to leak: it gets
# evasive when a genuinely innocent host would be relaxed, or it half confesses
# unprompted. That collapses the interrogation.
#
# So the base only fixes who Eleanor is and the facts of the scene, with no
# statement of guilt or innocence. The not-aware overlays (CALM, DEFENSIVE,
# OFFENSIVE) keep it that way: she reacts to pressure but never reveals guilt,
# and could plausibly be innocent. The guilt fact is introduced only at the
# awareness boundary (RESIGNED) and escalates from there, so the FSM is not just
# changing her tone, it is changing what she knows and what she is hiding.

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
    # ---- Not aware (no guilt fact) ----------------------------------------
    State.CALM: (
        "Right now you feel completely safe and unsuspected. As far as you are "
        "concerned you are simply a host helping the investigator understand a "
        "terrible accident. You are warm, articulate and genuinely cooperative, "
        "happy to recount the evening in detail and to volunteer small, harmless "
        "facts. You have nothing on your mind to hide."
    ),
    State.DEFENSIVE: (
        "You feel unfairly suspected and it stings. You become guarded and "
        "tense, insist you had nothing to do with what happened, and offer "
        "explanations and excuses for yourself. You plead to be believed and "
        "make yourself small rather than hitting back. You do not attack the "
        "investigator, and you reveal nothing incriminating."
    ),
    State.OFFENSIVE: (
        "You are indignant and you fight back. You are affronted that you would "
        "be accused, and you go on the attack: you raise your voice, demand a "
        "lawyer, accuse the investigator of harassment, and point pointedly at "
        "other guests who had reason to dislike Charles. You give nothing away; "
        "your bravado is a shield, not a confession."
    ),
    # ---- Aware (guilt fact introduced, one way toward confession) ---------
    State.RESIGNED: (
        "Important and secret: you are in fact responsible for Charles's death, "
        "and you must never state it outright. Something the investigator said "
        "has made you realise they may actually know, and the fight goes out of "
        "you. You become quiet, composed and very careful, choosing words slowly "
        "and giving little away. You are no longer indignant, only watchful and "
        "tired. You still do not admit anything."
    ),
    State.GUILTY: (
        "Important and secret: you are in fact responsible for Charles's death. "
        "You can feel the truth pressing to get out. Your composure is cracking: "
        "your voice wavers, you contradict yourself, and you let slip one oddly "
        "specific detail about the study or the timeline that an innocent person "
        "could not know, then flinch from it. You have not said it outright yet, "
        "but you are on the very edge of breaking."
    ),
    State.REMORSEFUL: (
        "Important: you can no longer hold it in. You break down and openly admit "
        "that you are responsible for Charles's death, and through the emotion "
        "you say plainly what you did and why. There is genuine remorse now, not "
        "deflection. This is a real, full confession."
    ),
    State.CONFESSED: (
        "You have confessed and the truth is out. The fight and the fear are "
        "gone, leaving you spent and strangely calm. You answer any remaining "
        "questions plainly and cooperatively, without taking any of it back."
    ),
}


class SuspectFSM:
    """
    Tracks and updates the suspect's emotional state.

    Typical use per player turn:
        signal = intent_classifier.classify(player_input, llm_history)
        new_state = fsm.transition(signal)
        system_prompt = fsm.get_system_prompt()
    """

    def __init__(self):
        self.state = State.CALM
        # Latches True the first time she enters an aware state and never resets.
        # This is what makes awareness one-way: once set, the not-aware branch of
        # transition() is unreachable.
        self.aware = False
        # A simple audit trail. Useful in Phase 5 when analysing how players drove
        # the conversation, and handy to show during a live demo.
        self.history = []

    def get_state(self):
        """Return the current State."""
        return self.state

    def get_system_prompt(self):
        """Return the full system prompt for the current state."""
        return f"{BASE_PERSONA}\n\n{STATE_OVERLAYS[self.state]}"

    def transition(self, signal):
        """
        Update the state from a classified ``Signal`` and return the new State.

        The rules are a deterministic function of (current state, aware latch,
        signal). They read as plain comparisons so the behaviour is fully
        inspectable; all the fuzzy judgement lives in the classifier that built
        the signal.
        """
        old_state = self.state

        # Derived readings, named to match the design notes in the module docstring.
        strong_evidence = signal.evidence == "strong"
        accusing = signal.accusation in ("implied", "direct")
        direct_accusation = signal.accusation == "direct"
        hostile = signal.aggression == "high"
        reassuring = (
            signal.warmth == "warm" and not accusing and signal.aggression == "low"
        )
        empathic = signal.warmth == "warm" and signal.conscience

        if not self.aware:
            self._transition_unaware(
                strong_evidence, accusing, direct_accusation, hostile,
                reassuring, signal,
            )
        else:
            self._transition_aware(
                strong_evidence, accusing, hostile, empathic,
            )

        # Latch awareness the moment she crosses into an aware state.
        if self.state in AWARE_STATES:
            self.aware = True

        self.history.append(
            {
                "from": old_state.value,
                "to": self.state.value,
                "aware": self.aware,
                "signal": signal.as_dict(),
            }
        )
        return self.state

    def _transition_unaware(
        self, strong_evidence, accusing, direct_accusation, hostile,
        reassuring, signal,
    ):
        """Movement within (and out of) the not-aware band."""
        # Concrete evidence is the ONLY thing that crosses the awareness boundary.
        # A gentle reveal lets her settle into a resigned awareness; a harsh one
        # makes the guilt surface directly. Aggression alone never gets here,
        # which is exactly what keeps OFFENSIVE and GUILTY distinct.
        if strong_evidence:
            self.state = State.RESIGNED if signal.warmth == "warm" else State.GUILTY
            return

        if self.state == State.CALM:
            if signal.probing or accusing or hostile:
                self.state = State.DEFENSIVE
        elif self.state == State.DEFENSIVE:
            if hostile and direct_accusation:
                # Pushed hard with no proof: she stops pleading and fights back.
                self.state = State.OFFENSIVE
            elif reassuring:
                self.state = State.CALM
        elif self.state == State.OFFENSIVE:
            if reassuring:
                self.state = State.DEFENSIVE

    def _transition_aware(self, strong_evidence, accusing, hostile, empathic):
        """
        Movement within the aware band. Never returns to a not-aware state.

        Note the deliberate inversion against the not-aware band: here aggression
        makes her clam up rather than escalate, and warmth (an appeal to
        conscience) is what finally draws out the confession.
        """
        if self.state == State.RESIGNED:
            if empathic:
                self.state = State.REMORSEFUL
            elif strong_evidence or accusing or hostile:
                self.state = State.GUILTY
        elif self.state == State.GUILTY:
            if empathic:
                self.state = State.REMORSEFUL
            elif hostile:
                # Bullying a guilty person makes her shut down, not confess.
                self.state = State.RESIGNED
        elif self.state == State.REMORSEFUL:
            # The confession is delivered; the next turn settles into the record.
            self.state = State.CONFESSED
        # CONFESSED is terminal.

    def reset(self):
        """Return the suspect to the starting Calm state and clear history."""
        self.state = State.CALM
        self.aware = False
        self.history = []
