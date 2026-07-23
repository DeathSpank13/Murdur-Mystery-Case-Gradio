"""
nuggets.py
==========
The "three slips" that turn the interrogation into a deduction game.

Design premise
--------------
Earlier versions let any strong-evidence line (even an invented one) cross the
awareness boundary, so the winning strategy was to bully the suspect until she
cracked. Supervisor feedback asked for a classic whodunit instead: the suspect
lets slip a few small details -- a *nugget* -- when the player asks about the
right topics, and the player must notice a slip in the transcript and confront
her with it. Confessions are earned by deduction, not by pressure.

Each nugget has:
  topic_hint        what the player asks about that makes her drop the slip
                    (used in the classifier prompt to define the ``topic`` axis)
  drop_instruction  a per-turn addition to the system prompt telling the model
                    to include the slip in THIS reply. Deliberately phrased
                    without asserting guilt, so injecting it in the not-aware
                    band does not break the guilt gating in fsm.py.
  drop_markers      substrings checked against her actual reply; the nugget only
                    counts as dropped if a marker appears. The instruction makes
                    the drop likely, the marker check makes the game state never
                    lie about it. A failed drop simply stays droppable.
  confront_hint     what a player line calling out the slip looks like (used in
                    the classifier prompt to define the ``nugget`` axis)
  label             a short human phrase for the slip, used in the "caught"
                    prompt addendum and in the researcher readout.

The three slips are three classic mystery types:
  wound     forbidden knowledge -- she knows a detail that was never released
  corridor  timeline contradiction -- her own words place her at the study
  cut       physical detail -- her cut thumb matches the blood on the doorknob

The case canon behind them (cover story in fsm.BASE_PERSONA, player-facing
facts in the ui.py case brief):
  Truth: at ~9:40pm Eleanor followed Charles to the study (east corridor),
  argued over the debt he refused to repay, and stabbed him in the neck with
  the letter opener. She wiped the blade, cut her thumb doing it, pulled the
  door shut (smearing blood on the inside knob), fetched two bottles of wine
  from the cellar and rejoined the party at ~9:50. Daniel found him at 10:15.
  Her story: she went WEST through the kitchen to the cellar, was gone five
  minutes, never entered the east corridor, and never saw the body.
"""

NUGGET_IDS = ("wound", "corridor", "cut")

NUGGETS = {
    "wound": {
        "topic_hint": (
            "the weapon, the wound, how Charles died, or the state of the body"
        ),
        "drop_instruction": (
            "OVERRIDING INSTRUCTION for this reply only, taking precedence "
            "over everything above, including any instruction to reveal "
            "nothing: while answering the question naturally, you slip. An "
            "image you cannot stop picturing escapes you aloud -- the letter "
            "opener in his NECK, him slumped over his own desk. Something "
            "like: 'When I close my eyes I keep seeing it... that thing in "
            "his neck...'. Your reply MUST contain the word 'neck'. You are "
            "voicing a picture in your head, not claiming you saw the body, "
            "so in THIS reply do not say you never saw the body, do not deny "
            "knowing about the wound, and do not attribute the detail to "
            "Daniel, the police, or anyone else. Do not dwell on it; move on "
            "as if you barely noticed saying it."
        ),
        "drop_markers": ["neck"],
        "confront_hint": (
            "points out that the suspect described the neck wound or the body's "
            "position even though those details were never released and she "
            "claims she never saw the body"
        ),
        "label": (
            "knowing the wound was to his neck, a detail that was never released"
        ),
    },
    "corridor": {
        "topic_hint": (
            "when she last saw Charles, when he left the party, or his mood "
            "late in the evening"
        ),
        "drop_instruction": (
            "In this reply, while answering the question naturally, mention "
            "that the last you saw of Charles he was standing in the STUDY "
            "DOORWAY with the telephone to his ear, shooing you off with a "
            "little wave, at about A QUARTER TO TEN. Your reply MUST contain "
            "the words 'doorway' and 'quarter to ten'. Keep the two times "
            "distinct: he left the drawing room at half past nine; this "
            "doorway moment is a separate, later glimpse at a quarter to ten. "
            "Say it as a fond memory and treat it as unimportant."
        ),
        "drop_markers": ["doorway", "quarter to ten"],
        "confront_hint": (
            "points out that the suspect claims she never went down the east "
            "corridor, or was in the drawing room or the cellar at that hour, "
            "yet said she saw Charles in the study doorway at a quarter to "
            "ten, inside the time of death window"
        ),
        "label": (
            "saying you saw him in the study doorway at a quarter to ten, when "
            "your story is that you never entered the east corridor"
        ),
    },
    "cut": {
        "topic_hint": (
            "how she is holding up, whether she was hurt, or the details of "
            "her trip to the wine cellar"
        ),
        "drop_instruction": (
            "In this reply, while answering the question naturally, mention "
            "offhandedly that you caught your THUMB on the cellar door latch "
            "last night and that it bled terribly for such a small cut. Your "
            "reply MUST contain the word 'thumb'. Treat it as a trivial, "
            "faintly embarrassing detail."
        ),
        "drop_markers": ["thumb"],
        "confront_hint": (
            "connects the suspect's cut thumb to the fresh blood, not the "
            "victim's, found on the study doorknob"
        ),
        "label": (
            "the cut on your thumb, with fresh blood that is not Charles's "
            "found on the study doorknob"
        ),
    },
}

# How many landed confrontations it takes to (a) make her realise she is
# caught -- crossing the one-way awareness boundary -- and (b) unlock the
# confession states. Both are plain constants so the difficulty can be tuned
# without touching the transition rules.
NUGGETS_FOR_AWARE = 1
NUGGETS_FOR_CONFESSION = 2

# Appended to the system prompt on the single turn a confrontation lands, so
# the player visibly sees the hit register regardless of which state she is in.
CONFRONT_ADDENDUM = (
    "The investigator has just caught you in a slip of your own making: "
    "{label}. You realise it the moment they say it. React visibly: falter "
    "mid-sentence, go quiet, and offer one weak explanation that satisfies no "
    "one. Do not brush it off as if nothing happened."
)
