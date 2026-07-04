"""
branching_dialogue.py
=====================
A choice-based ("dialogue tree") version of the interrogation, in the style
traditional games use: the player picks from a menu of options instead of typing
free text. It is a separate mode from the keyword-matched `static_dialogue.py`
and does not touch the blinded A/B study.

What makes it more than a flat list of buttons is the three consequence rules,
all enforced by the engine when it decides which options are visible:

    once             A question that can only be asked once. After it is chosen
                     it is consumed and never offered again this run.
    exclusive_group  A fork in approach. Options sharing a group are mutually
                     exclusive: choosing one locks the whole group, so the
                     player never sees the sibling paths they passed up.
    goto / nesting   Choosing a topic descends into a child node of follow-up
                     sub-questions. The once/exclusive rules apply at every
                     level, and a synthetic "Back" option climbs back out.

The dialogue lives in DIALOGUE_TREE as plain data; DialogueEngine walks it and
tracks per-run state (current node, consumed questions, locked groups, a
navigation stack, and the transcript). The engine is UI-agnostic so it can be
unit-tested without launching Gradio (see test_dialogue.py).
"""

# Identifier of the synthetic option the engine offers to climb one level back
# up the tree. It never appears in DIALOGUE_TREE; the engine injects it.
BACK_ID = "__back__"
BACK_TEXT = "← Back"

# How many substantive questions the player must ask before the accusation
# options (guilty / innocent) unlock at the root menu.
ACCUSE_AFTER = 3


# ---------------------------------------------------------------------------
# The dialogue tree
# ---------------------------------------------------------------------------
# Each node has an "intro" (what Eleanor says when the player arrives there) and
# an ordered list of "options". An option is a dict with:
#   id               unique string, used to track consumption and navigation
#   text             the player-facing line (the button label)
#   response         Eleanor's reply when the option is chosen (optional)
#   once             True -> consumed after first use (a one-time question)
#   exclusive_group  name -> choosing it locks every option sharing that name
#   goto             node id -> descend into that node (a nested follow-up)
#   min_questions    int -> hidden until that many questions have been asked
#   accusation       "guilty"/"innocent" -> a verdict option; picking it does
#                    not count toward the question total, so it can't unlock
#                    itself, and it reveals whether the accusation was correct
#
# The scenario matches the rest of the project: the player questions Eleanor
# Vance about Charles Whitmore's death at her dinner party.

DIALOGUE_TREE = {
    "main": {
        "intro": (
            "Of course, Inspector. Ask me whatever you need. It has been a "
            "dreadful day, but I'd like to help you make sense of it. Where "
            "shall we start?"
        ),
        "options": [
            {
                "id": "party",
                "text": "Tell me about last night's party.",
                "response": (
                    "A small gathering. Old friends, a little wine, far too "
                    "much talk of business. Charles was in good spirits when he "
                    "arrived."
                ),
                "goto": "party",
            },
            {
                "id": "alibi",
                "text": "Where were you when Charles died?",
                "response": (
                    "In the drawing room with my guests, for most of the "
                    "evening. I stepped out once for wine, no more than a few "
                    "minutes. Ask any of them."
                ),
                "once": True,
                "goto": "alibi",
            },
            {
                "id": "relationship",
                "text": "What was your relationship with Charles?",
                "response": (
                    "We went back twenty years. Friends, and partners in a "
                    "gallery or two. Friendship like that has its weather, "
                    "Inspector."
                ),
                "goto": "relationship",
            },
            {
                "id": "weapon",
                "text": "Let's talk about the murder weapon.",
                "response": (
                    "The letter opener from the study, I'm told. A dreadful "
                    "thing to imagine. It sat on the desk in plain view of "
                    "anyone who passed."
                ),
                "goto": "weapon",
            },
            # The approach fork: pick one tone and live with it. Choosing either
            # locks the other away for the rest of the run, so the player never
            # gets to try both on the same suspect.
            {
                "id": "press",
                "text": "I think you're hiding something. (press her hard)",
                "response": (
                    "Hiding something? I open my home, lose a friend, and am "
                    "rewarded with accusations. Tread carefully, Inspector."
                ),
                "exclusive_group": "approach",
                "goto": "press",
            },
            {
                "id": "reassure",
                "text": "You're not a suspect. Help me understand. (reassure her)",
                "response": (
                    "Thank you. It has been a horrid day, and to be treated "
                    "gently is a kindness. Ask me anything; I want him found "
                    "out as much as you do."
                ),
                "exclusive_group": "approach",
                "goto": "reassure",
            },
            # The verdict. Hidden until the player has asked a few questions, then
            # offered as two mutually exclusive accusations. Choosing one locks
            # the other and reveals whether the call was right; the conversation
            # may continue afterwards.
            {
                "id": "verdict_guilty",
                "text": "Eleanor Vance, I'm arresting you for Charles's murder.",
                "response": (
                    "Her composure finally cracks. \"You have no idea what he "
                    "meant to take from me.\" She was, in fact, responsible for "
                    "Charles's death. Your accusation was correct."
                ),
                "min_questions": ACCUSE_AFTER,
                "once": True,
                "exclusive_group": "verdict",
                "accusation": "guilty",
            },
            {
                "id": "verdict_innocent",
                "text": "I don't believe you did it. You're free to go.",
                "response": (
                    "\"Thank you, Inspector. You're wiser than you look.\" She "
                    "was, in fact, responsible for Charles's death. Your "
                    "accusation was incorrect."
                ),
                "min_questions": ACCUSE_AFTER,
                "once": True,
                "exclusive_group": "verdict",
                "accusation": "innocent",
            },
        ],
    },

    "party": {
        "intro": "What is it about the evening you'd like to know?",
        "options": [
            {
                "id": "guests",
                "text": "Who else was at the party?",
                "response": (
                    "The Harringtons, my business partner Vivian, Charles, and "
                    "young Daniel who keeps my books. Seven of us, with the "
                    "staff."
                ),
            },
            {
                "id": "argument",
                "text": "Did anyone argue with Charles that night?",
                "response": (
                    "Daniel and he had words over money near the end. Quiet, "
                    "but I saw Daniel's face. I'd not make too much of it, "
                    "though. He is a gentle boy."
                ),
                "once": True,
            },
            {
                "id": "lastseen",
                "text": "When did you last see Charles alive?",
                "response": (
                    "When he left the drawing room to take a telephone call, "
                    "half past nine or so. I went to see to the wine a little "
                    "later and never spoke to him again."
                ),
                "once": True,
            },
        ],
    },

    "alibi": {
        "intro": "My whereabouts. Press me on it if you must.",
        "options": [
            {
                "id": "wine_cellar",
                "text": "Tell me about the trip to the cellar.",
                "response": (
                    "Down the back stairs, two bottles of the Margaux, back up. "
                    "Five minutes, perhaps seven. The cellar is cold; one "
                    "doesn't linger."
                ),
                "once": True,
            },
            {
                "id": "witness",
                "text": "Can anyone confirm you were in the drawing room?",
                "response": (
                    "Vivian, certainly. She and I were thick as thieves on the "
                    "settee most of the night. Though she did step out for some "
                    "air around then, now I think of it."
                ),
            },
        ],
    },

    "relationship": {
        "intro": "Charles and I. Where shall I start?",
        "options": [
            {
                "id": "business",
                "text": "You did business together?",
                "response": (
                    "Three galleries over the years. The last one did poorly, "
                    "and money has a way of souring even old affection."
                ),
                "goto": "business",
            },
            {
                "id": "disagreement",
                "text": "You mentioned disagreements. About what?",
                "response": (
                    "The usual things. He thought me reckless with the "
                    "accounts; I thought him a coward with them. We were both a "
                    "little right."
                ),
                "once": True,
            },
        ],
    },

    "business": {
        "intro": "The business, then. It's no secret it ended badly.",
        "options": [
            {
                "id": "debt",
                "text": "Did Charles owe you money, or you him?",
                "response": (
                    "He owed me. A great deal, and he was slow about it. I'd "
                    "have been paid eventually. I am not a fool about these "
                    "things."
                ),
                "once": True,
            },
            {
                "id": "insurance",
                "text": "Was there any insurance or payout tied to him?",
                "response": (
                    "On the gallery partnership, yes, a modest one. Standard "
                    "between partners. I'd hardly call it a fortune."
                ),
                "once": True,
            },
        ],
    },

    "weapon": {
        "intro": "That wretched letter opener. What of it?",
        "options": [
            {
                "id": "who_handled",
                "text": "Who could have handled it?",
                "response": (
                    "Anyone. It lived on the study desk. I'd not touched it in "
                    "weeks. It was decorative more than useful."
                ),
            },
            {
                "id": "prints",
                "text": "Whose fingerprints would we expect to find on it?",
                "response": (
                    "Mine, I suppose, from dusting the desk. And half the "
                    "county's, for all I know. It's hardly under lock and key."
                ),
                "once": True,
            },
        ],
    },

    # The two approach branches. Each is a real, different conversation, and the
    # player can only ever walk down one of them per run.
    "press": {
        "intro": (
            "Go on, then. Bully me with your theories and see where it gets "
            "you."
        ),
        "options": [
            {
                "id": "accuse_direct",
                "text": "You killed Charles, didn't you?",
                "response": (
                    "How dare you. I want my solicitor, and I want this "
                    "conversation noted, every word of it. I'll not say another "
                    "thing without counsel."
                ),
                "once": True,
            },
            {
                "id": "motive_money",
                "text": "He owed you money. That's a motive.",
                "response": (
                    "A debt is a reason to keep a man alive and paying, "
                    "Inspector, not to put a blade in him. Do think it "
                    "through."
                ),
                "once": True,
            },
        ],
    },

    "reassure": {
        "intro": (
            "You're kind to say so. What can I tell you that would help?"
        ),
        "options": [
            {
                "id": "who_suspect",
                "text": "Who do you think could have done this?",
                "response": (
                    "If I had to point a finger, and I hate to do it, I would "
                    "look at Daniel and that quarrel over money. But I may be "
                    "wronging the boy."
                ),
                "once": True,
            },
            {
                "id": "anything_odd",
                "text": "Did anything seem out of place that night?",
                "response": (
                    "The study door was shut when it's always left open. I "
                    "noticed it and thought nothing of it. Perhaps I should "
                    "have."
                ),
                "once": True,
            },
        ],
    },
}


class DialogueEngine:
    """
    Walks DIALOGUE_TREE for one run and tracks the consequence state.

    Typical use per player turn:
        for opt in engine.available_options():
            ... render a button ...
        engine.choose(chosen_option_id)   # advances state + transcript

    The transcript is a list of (speaker, text) pairs, where speaker is
    "npc" or "player", seeded with the start node's intro line.
    """

    def __init__(self, tree=None, start="main"):
        self.tree = tree if tree is not None else DIALOGUE_TREE
        self.current = start
        self.consumed = set()        # ids of one-time options already used
        self.locked_groups = set()   # exclusive groups already committed to
        self.chosen = set()          # ids of every option ever picked (for the
                                     # UI to dim repeatable options already used)
        self.stack = []              # node ids to climb back through
        self.transcript = []         # [(speaker, text), ...]
        self.questions_asked = 0     # substantive questions asked (gates verdicts)

        intro = self.tree[start].get("intro")
        if intro:
            self.transcript.append(("npc", intro))

    def available_options(self):
        """
        Return the ordered options visible at the current node right now.

        Drops one-time options that have been used and options whose exclusive
        group is already locked. Appends a synthetic Back option when there is
        somewhere to climb back to.
        """
        options = []
        for opt in self.tree[self.current]["options"]:
            if opt.get("once") and opt["id"] in self.consumed:
                continue
            group = opt.get("exclusive_group")
            if group and group in self.locked_groups:
                continue
            if opt.get("min_questions") and self.questions_asked < opt["min_questions"]:
                continue
            options.append(opt)

        if self.stack:
            options.append({"id": BACK_ID, "text": BACK_TEXT})
        return options

    def choose(self, option_id):
        """
        Apply the option with the given id and return it.

        Updates the transcript, marks one-time questions consumed, locks
        exclusive groups, and navigates (descend on `goto`, climb on Back).
        Raises KeyError if the id is not currently available, so UI bugs that
        send a stale id surface loudly instead of silently misbehaving.
        """
        if option_id == BACK_ID:
            if self.stack:
                self.current = self.stack.pop()
                intro = self.tree[self.current].get("intro")
                if intro:
                    self.transcript.append(("npc", intro))
            return {"id": BACK_ID, "text": BACK_TEXT}

        option = self._find_available(option_id)

        self.transcript.append(("player", option["text"]))
        if option.get("response"):
            self.transcript.append(("npc", option["response"]))

        # Remember every pick. One-time/exclusive options leave the menu, but
        # repeatable ones stay on offer, and the UI dims those once chosen.
        self.chosen.add(option_id)
        if option.get("once"):
            self.consumed.add(option_id)
        group = option.get("exclusive_group")
        if group:
            self.locked_groups.add(group)
        # Real questions advance the counter that unlocks the accusations; the
        # accusations themselves don't, so one can never satisfy its own gate.
        if not option.get("accusation"):
            self.questions_asked += 1

        target = option.get("goto")
        if target:
            self.stack.append(self.current)
            self.current = target
            intro = self.tree[target].get("intro")
            if intro:
                self.transcript.append(("npc", intro))

        return option

    def _find_available(self, option_id):
        for opt in self.available_options():
            if opt["id"] == option_id:
                return opt
        raise KeyError(
            f"Option {option_id!r} is not available at node {self.current!r}."
        )
