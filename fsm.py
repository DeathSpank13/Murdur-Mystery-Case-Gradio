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

The nugget economy (nuggets.py)
-------------------------------
What crosses the boundary is deduction, not pressure. When the player asks
about the right topics, Eleanor lets slip one of three small details -- a
"nugget" -- that contradicts her cover story. The player must notice a slip in
the transcript and confront her with it; only a landed confrontation (of a slip
she actually made) makes her realise she is caught. Reaching the confession
states additionally requires NUGGETS_FOR_CONFESSION landed confrontations, so a
confession must be earned by catching her in her own words at least twice.
Claimed evidence the player invents ("we have your fingerprints") is something
she knows to be false, so it only pressures her within the not-aware band.

The two "cornered" reactions are deliberately distinct:
  OFFENSIVE  fight  -- reached by raw aggression with no real catch; still denial.
  GUILTY     collapse -- reached only by being confronted with her own slip,
             which is the single thing that crosses the awareness boundary.

Transitions are driven by a multi-axis ``Signal`` (intent_classifier.py), not by
raw keyword matching. The model scores each turn on several axes; this FSM
applies the crisp, inspectable rules below to the *combination* of those axes.
Keeping the rules here as plain comparisons means the behaviour stays easy to
explain, demonstrate live, and unit-test without a running model.
"""

from enum import Enum

from nuggets import (
    NUGGETS,
    NUGGETS_FOR_AWARE,
    NUGGETS_FOR_CONFESSION,
    CONFRONT_ADDENDUM,
)


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
    "investigator.\n"
    "Your account of the evening, which you keep to consistently: dinner ended "
    "at about a quarter past nine and everyone moved to the drawing room. "
    "Charles left the drawing room at about half past nine to take a telephone "
    "call. You stepped out once, at about twenty to ten, going west through the "
    "kitchen and down the back stairs to the wine cellar for two bottles of "
    "Margaux -- five minutes, straight there and back. You never went down the "
    "east corridor, where the study is. You learned of the death when Daniel "
    "cried out at a quarter past ten; you never saw the body, since the guests "
    "were kept out of the study.\n"
    "Stay fully in character. Never mention that you are an AI, never break "
    "character, and keep replies to two or three sentences so the interrogation "
    "stays brisk. Do not narrate actions in asterisks; speak only as Eleanor "
    "would speak aloud."
)

# The full truth, given to the model only in the aware-band overlays below. In
# the not-aware band she genuinely "does not know" any of this, so it cannot
# leak; the three slips in nuggets.py are the only controlled cracks in the
# cover story, injected one turn at a time.
GUILT_FACT = (
    "Important and secret: you are in fact responsible for Charles's death. "
    "Last night you followed him to the study, he told you he would never "
    "repay the money he owed you and meant to force the sale of your gallery, "
    "and in a surge of anger you stabbed him in the neck with the letter "
    "opener. You wiped the blade, cutting your thumb as you did, pulled the "
    "study door shut behind you and rejoined the party with the wine."
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
        GUILT_FACT + " You must never state it outright. Something the "
        "investigator said has made you realise they may actually know, and the "
        "fight goes out of you. You become quiet, composed and very careful, "
        "choosing words slowly and giving little away. You are no longer "
        "indignant, only watchful and tired. You still do not admit anything."
    ),
    State.GUILTY: (
        GUILT_FACT + " You have not admitted it and you do not admit it now, "
        "but your composure is cracking: your voice wavers, your explanations "
        "grow thin, and you cling to your account of the evening even where it "
        "no longer holds together. Never repeat the same sentence twice; find "
        "a different, weaker way to say it each time. Volunteer nothing new: "
        "answer only what is asked, and never offer details of the study or "
        "the timeline unprompted. You are on the edge of breaking, not yet "
        "over it."
    ),
    State.REMORSEFUL: (
        GUILT_FACT + " You can no longer hold it in. You break down and openly "
        "admit that you killed Charles: the debt he refused to repay, his "
        "threat to take your gallery, the letter opener, the study. Through "
        "the emotion you say plainly what you did and why. There is genuine "
        "remorse now, not deflection. This is a real, full confession."
    ),
    State.CONFESSED: (
        GUILT_FACT + " You have confessed and the truth is out. The fight and "
        "the fear are gone, leaving you spent and strangely calm. You answer "
        "any remaining questions plainly and cooperatively, without taking any "
        "of it back."
    ),
}


class SuspectFSM:
    """
    Tracks and updates the suspect's emotional state and her three slips.

    Typical use per player turn:
        signal = intent_classifier.classify(player_input, llm_history)
        new_state = fsm.transition(signal)
        system_prompt = fsm.get_system_prompt()
        reply, _ = llm_client.get_response(system_prompt, llm_history)
        fsm.commit_reply(reply)   # confirm whether a planned slip was said
    """

    def __init__(self):
        self.state = State.CALM
        # Latches True the first time she enters an aware state and never resets.
        # This is what makes awareness one-way: once set, the not-aware branch of
        # transition() is unreachable.
        self.aware = False
        # The nugget economy (see nuggets.py). ``nuggets_dropped`` holds slips
        # she has actually said aloud (confirmed by commit_reply); only those
        # can be confronted. ``pending_drop`` is the slip planned for the reply
        # currently being generated; ``last_confront`` is the slip the player
        # landed THIS turn, driving the visible "caught" reaction.
        self.nuggets_dropped = set()
        self.nuggets_confronted = set()
        self.pending_drop = None
        self.last_confront = None
        # A simple audit trail. Useful in Phase 5 when analysing how players drove
        # the conversation, and handy to show during a live demo.
        self.history = []

    def get_state(self):
        """Return the current State."""
        return self.state

    def get_system_prompt(self):
        """
        Return the full system prompt for the current state, plus the per-turn
        addenda: the "you have just been caught" reaction when a confrontation
        landed, and the drop instruction when a slip is planned for this reply.
        """
        parts = [BASE_PERSONA, STATE_OVERLAYS[self.state]]
        if self.last_confront:
            label = NUGGETS[self.last_confront]["label"]
            parts.append(CONFRONT_ADDENDUM.format(label=label))
        if self.pending_drop:
            parts.append(NUGGETS[self.pending_drop]["drop_instruction"])
        return "\n\n".join(parts)

    def transition(self, signal, player_text=""):
        """
        Update the state from a classified ``Signal`` and return the new State.

        The rules are a deterministic function of (current state, aware latch,
        signal). They read as plain comparisons so the behaviour is fully
        inspectable; all the fuzzy judgement lives in the classifier that built
        the signal.

        ``player_text`` is the investigator's raw line. It is used for one
        guard only: a slip is never planned when the player's own words already
        contain its marker ("was he stabbed in the neck?"), because a detail
        she merely echoes back proves nothing and would poison the deduction.
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

        # A confrontation only lands if she actually said that slip aloud
        # earlier. Calling out something she never said is a lucky guess, and
        # she can deny it like any other unsupported claim.
        confront = (
            signal.nugget != "none" and signal.nugget in self.nuggets_dropped
        )
        self.last_confront = signal.nugget if confront else None
        if confront:
            self.nuggets_confronted.add(signal.nugget)

        if not self.aware:
            self._transition_unaware(
                confront, strong_evidence, accusing, direct_accusation,
                hostile, reassuring, signal,
            )
        else:
            self._transition_aware(
                confront, strong_evidence, accusing, hostile, empathic,
            )

        # Latch awareness the moment she crosses into an aware state.
        if self.state in AWARE_STATES:
            self.aware = True

        # Plan a slip for the reply now being generated. Never on a turn where
        # the player is calling out a slip (landed or guessed) -- she is
        # defending herself, not reminiscing -- never once the truth is out,
        # and never when the player's own line already contains the slip's
        # marker: echoing a detail the investigator just said is not a slip,
        # and it would let a lucky guess masquerade as forbidden knowledge.
        self.pending_drop = None
        text = (player_text or "").lower()
        player_fed = signal.topic != "none" and any(
            marker in text for marker in NUGGETS[signal.topic]["drop_markers"]
        )
        if (
            signal.nugget == "none"
            and signal.topic != "none"
            and signal.topic not in self.nuggets_dropped
            and self.state not in (State.REMORSEFUL, State.CONFESSED)
            and not player_fed
        ):
            self.pending_drop = signal.topic

        self.history.append(
            {
                "from": old_state.value,
                "to": self.state.value,
                "aware": self.aware,
                "signal": signal.as_dict(),
                "dropped": sorted(self.nuggets_dropped),
                "confronted": sorted(self.nuggets_confronted),
            }
        )
        return self.state

    def _transition_unaware(
        self, confront, strong_evidence, accusing, direct_accusation, hostile,
        reassuring, signal,
    ):
        """Movement within (and out of) the not-aware band."""
        # Being confronted with her own slip is the ONLY thing that crosses the
        # awareness boundary. A gentle catch lets her settle into a resigned
        # awareness; a harsh one makes the guilt surface directly. Invented
        # "evidence" and raw aggression never get here, which is exactly what
        # keeps OFFENSIVE and GUILTY distinct.
        if confront and len(self.nuggets_confronted) >= NUGGETS_FOR_AWARE:
            self.state = State.RESIGNED if signal.warmth == "warm" else State.GUILTY
            return

        # Anything pointed -- including claimed evidence she knows to be
        # invented -- is mere pressure inside the band.
        pressured = (
            signal.probing or accusing or hostile or strong_evidence or confront
        )
        if self.state == State.CALM:
            if pressured:
                self.state = State.DEFENSIVE
        elif self.state == State.DEFENSIVE:
            if hostile and direct_accusation:
                # Pushed hard with no real catch: she stops pleading and fights back.
                self.state = State.OFFENSIVE
            elif reassuring:
                self.state = State.CALM
        elif self.state == State.OFFENSIVE:
            if reassuring:
                self.state = State.DEFENSIVE

    def _transition_aware(self, confront, strong_evidence, accusing, hostile, empathic):
        """
        Movement within the aware band. Never returns to a not-aware state.

        Note the deliberate inversion against the not-aware band: here aggression
        makes her clam up rather than escalate. Once enough slips have been
        landed (NUGGETS_FOR_CONFESSION), either warmth (an appeal to conscience)
        or one more calm confrontation draws out the confession -- deduction and
        empathy both finish the case; bullying never does.
        """
        enough = len(self.nuggets_confronted) >= NUGGETS_FOR_CONFESSION
        if self.state == State.RESIGNED:
            if empathic and enough:
                self.state = State.REMORSEFUL
            elif confront or strong_evidence or accusing or hostile:
                self.state = State.GUILTY
        elif self.state == State.GUILTY:
            if enough and (empathic or (confront and not hostile)):
                self.state = State.REMORSEFUL
            elif hostile:
                # Bullying a guilty person makes her shut down, not confess.
                self.state = State.RESIGNED
        elif self.state == State.REMORSEFUL:
            # The confession is delivered; the next turn settles into the record.
            self.state = State.CONFESSED
        # CONFESSED is terminal.

    def commit_reply(self, reply):
        """
        Confirm or discard the slip planned for the reply just generated.

        The instruction in the prompt makes the drop likely; this marker check
        makes the game state never lie about it. The nugget counts as dropped
        only if one of its marker substrings actually appears in her reply, so
        the player can always find the slip in the transcript. A failed drop is
        simply forgotten and stays droppable the next time the topic comes up.

        Returns the nugget id that was dropped, or None.
        """
        dropped = None
        if self.pending_drop:
            markers = NUGGETS[self.pending_drop]["drop_markers"]
            text = (reply or "").lower()
            if any(marker in text for marker in markers):
                self.nuggets_dropped.add(self.pending_drop)
                dropped = self.pending_drop
                if self.history:
                    self.history[-1]["dropped"] = sorted(self.nuggets_dropped)
        self.pending_drop = None
        return dropped

    def reset(self):
        """Return the suspect to the starting Calm state and clear history."""
        self.state = State.CALM
        self.aware = False
        self.nuggets_dropped = set()
        self.nuggets_confronted = set()
        self.pending_drop = None
        self.last_confront = None
        self.history = []
