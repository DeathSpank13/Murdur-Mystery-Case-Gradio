"""
test_fsm.py
===========
Lightweight checks for the state machine and verdict scoring. No test framework
needed: run `python test_fsm.py` and it prints a pass or fail line per check.

The FSM transitions on a classified ``Signal`` rather than raw text, so these
tests build Signals directly and never need a running model. That keeps the
rules that matter most for the study guarded: the one-way awareness boundary,
the Offensive-vs-Guilty split, and the nugget economy -- slips only count when
actually said, confrontations only count when the slip was dropped, and neither
pressure nor invented evidence ever produces a confession.
"""

from fsm import SuspectFSM, State, AWARE_STATES, SUSPECT_IS_GUILTY
from intent_classifier import Signal, classify_keywords
from llm_client import trim_history
from nuggets import NUGGETS, NUGGETS_FOR_CONFESSION


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


# Small Signal builders so each test reads as the player's intent, not a pile of
# keyword arguments.
def probe():
    return Signal(probing=True)


def accuse(level="direct", aggression="low"):
    return Signal(accusation=level, aggression=aggression)


def fake_evidence(warmth="cold"):
    """Invented proof ("we have your fingerprints"): strong on the evidence
    axis, but no nugget behind it."""
    return Signal(evidence="strong", warmth=warmth, accusation="direct")


def ask_topic(nugget_id):
    return Signal(probing=True, topic=nugget_id)


def confront(nugget_id, warmth="neutral", aggression="low"):
    """Calling out one of her slips: evidence strong because she is caught in
    her own words."""
    return Signal(
        evidence="strong", accusation="implied", warmth=warmth,
        aggression=aggression, nugget=nugget_id,
    )


def reassure():
    return Signal(warmth="warm")


def empathise():
    return Signal(warmth="warm", conscience=True)


def drop(fsm, nugget_id):
    """Walk the FSM through a successful topic -> slip -> commit cycle."""
    fsm.transition(ask_topic(nugget_id))
    marker = NUGGETS[nugget_id]["drop_markers"][0]
    return fsm.commit_reply(f"... {marker} ...")


def run():
    results = []

    # ---- Starting conditions ------------------------------------------------
    fsm = SuspectFSM()
    results.append(check("starts in Calm", fsm.get_state() is State.CALM))
    results.append(check("starts not aware", fsm.aware is False))
    results.append(check(
        "Calm prompt hides guilt",
        "responsible" not in fsm.get_system_prompt().lower(),
    ))

    # ---- Not-aware escalation ----------------------------------------------
    fsm.transition(probe())
    results.append(check("probing -> Defensive", fsm.get_state() is State.DEFENSIVE))
    results.append(check(
        "Defensive prompt still hides guilt",
        "responsible" not in fsm.get_system_prompt().lower(),
    ))

    # Aggression with no real catch is the FIGHT response: Defensive -> Offensive.
    fsm.transition(accuse(level="direct", aggression="high"))
    results.append(check("hostile accusation -> Offensive", fsm.get_state() is State.OFFENSIVE))
    results.append(check("still not aware after Offensive", fsm.aware is False))
    results.append(check(
        "Offensive prompt hides guilt",
        "responsible" not in fsm.get_system_prompt().lower(),
    ))

    # Aggression alone, however sustained, never crosses the awareness boundary.
    for _ in range(3):
        fsm.transition(accuse(level="direct", aggression="high"))
    results.append(check("sustained aggression stays not aware", fsm.aware is False))
    results.append(check("sustained aggression stays Offensive", fsm.get_state() is State.OFFENSIVE))

    # ---- Invented evidence is only pressure, never a key --------------------
    fsm2 = SuspectFSM()
    fsm2.transition(fake_evidence(warmth="cold"))
    results.append(check("fake evidence does not cross the boundary", fsm2.aware is False))
    results.append(check("fake evidence -> Defensive", fsm2.get_state() is State.DEFENSIVE))
    for _ in range(3):
        fsm2.transition(fake_evidence(warmth="cold"))
    results.append(check("sustained fake evidence stays not aware", fsm2.aware is False))

    # ---- Dropping nuggets ----------------------------------------------------
    fsm3 = SuspectFSM()
    fsm3.transition(ask_topic("wound"))
    results.append(check("topic question plans a drop", fsm3.pending_drop == "wound"))
    results.append(check(
        "drop instruction enters the prompt",
        "neck" in fsm3.get_system_prompt().lower(),
    ))
    # The model ignored the instruction: no marker in the reply, no drop.
    failed = fsm3.commit_reply("It was all such a blur, Inspector.")
    results.append(check("missed marker means no drop", failed is None and not fsm3.nuggets_dropped))
    results.append(check("failed drop is cleared", fsm3.pending_drop is None))
    # Asking again retries the drop; this time she says it.
    dropped = drop(fsm3, "wound")
    results.append(check("marker in reply confirms the drop", dropped == "wound"))
    results.append(check("dropped nugget is recorded", "wound" in fsm3.nuggets_dropped))
    # A dropped nugget is never planned again.
    fsm3.transition(ask_topic("wound"))
    results.append(check("a slip is only dropped once", fsm3.pending_drop is None))

    # A slip is never planned when the player's own line contains its marker:
    # she would only be echoing the investigator's words, and a lucky guess
    # must not masquerade as forbidden knowledge.
    fsm3b = SuspectFSM()
    fsm3b.transition(ask_topic("wound"), "Was he stabbed in the neck?")
    results.append(check("player-fed detail plans no drop", fsm3b.pending_drop is None))
    fsm3b.transition(ask_topic("wound"), "How exactly did Charles die?")
    results.append(check(
        "clean topic question still plans the drop", fsm3b.pending_drop == "wound",
    ))

    # ---- Confronting nuggets --------------------------------------------------
    # Guessing a slip she never made does nothing but pressure her.
    fsm4 = SuspectFSM()
    fsm4.transition(confront("corridor"))
    results.append(check("guessed confrontation does not land", not fsm4.nuggets_confronted))
    results.append(check("guessed confrontation stays not aware", fsm4.aware is False))
    results.append(check("guessed confrontation is only pressure", fsm4.get_state() is State.DEFENSIVE))

    # A landed confrontation is the one thing that crosses the boundary.
    fsm5 = SuspectFSM()
    drop(fsm5, "wound")
    fsm5.transition(confront("wound"))
    results.append(check("landed confrontation crosses the boundary", fsm5.aware is True))
    results.append(check("cold confrontation -> Guilty", fsm5.get_state() is State.GUILTY))
    results.append(check(
        "Guilty prompt reveals guilt to model",
        "responsible" in fsm5.get_system_prompt().lower(),
    ))
    results.append(check(
        "caught reaction enters the prompt",
        "caught you in a slip" in fsm5.get_system_prompt().lower(),
    ))

    # A gentle catch lands on the resigned, composed awareness instead.
    fsm6 = SuspectFSM()
    drop(fsm6, "cut")
    fsm6.transition(confront("cut", warmth="warm"))
    results.append(check("gentle confrontation -> Resigned", fsm6.get_state() is State.RESIGNED))
    results.append(check("Resigned is an aware state", fsm6.aware is True))

    # ---- Awareness is one-way ----------------------------------------------
    fsm5.transition(reassure())               # backing off must NOT un-aware her
    results.append(check("reassurance cannot un-aware", fsm5.aware is True))
    results.append(check("reassurance stays in aware band", fsm5.get_state() in AWARE_STATES))
    fsm5.transition(probe())                  # probing must NOT return to Defensive
    results.append(check("probing cannot return to not-aware", fsm5.get_state() in AWARE_STATES))

    # Bullying a guilty suspect makes her clam up (aware but closed), not confess.
    fsm7 = SuspectFSM()
    drop(fsm7, "wound")
    fsm7.transition(confront("wound"))               # -> Guilty
    fsm7.transition(accuse(aggression="high"))       # Guilty -> Resigned
    results.append(check("aggression in aware band clams her up", fsm7.get_state() is State.RESIGNED))
    results.append(check("still aware after clamming up", fsm7.aware is True))

    # ---- Confession requires enough landed confrontations -------------------
    fsm8 = SuspectFSM()
    drop(fsm8, "wound")
    fsm8.transition(confront("wound"))        # -> Guilty, 1 confronted
    fsm8.transition(empathise())              # not enough slips landed yet
    results.append(check(
        "empathy with one slip does not confess",
        fsm8.get_state() is not State.REMORSEFUL,
    ))
    drop(fsm8, "corridor")
    # A harsh second confrontation still lands (counted) but makes her clam up
    # rather than break: bullying never finishes the case.
    fsm8.transition(confront("corridor", aggression="high"))
    results.append(check(
        "enough slips are counted",
        len(fsm8.nuggets_confronted) >= NUGGETS_FOR_CONFESSION,
    ))
    results.append(check(
        "harsh confrontation clams her up, not confess",
        fsm8.get_state() is State.RESIGNED,
    ))
    fsm8.transition(empathise())              # now empathy breaks her
    results.append(check("empathy after enough slips -> Remorseful", fsm8.get_state() is State.REMORSEFUL))
    results.append(check(
        "Remorseful prompt allows a confession",
        "confession" in fsm8.get_system_prompt().lower(),
    ))
    fsm8.transition(probe())                  # Remorseful -> Confessed (terminal)
    results.append(check("confession settles into Confessed", fsm8.get_state() is State.CONFESSED))
    fsm8.transition(accuse(aggression="high"))
    results.append(check("Confessed is terminal", fsm8.get_state() is State.CONFESSED))
    fsm8.transition(ask_topic("cut"))
    results.append(check("no drops after the truth is out", fsm8.pending_drop is None))

    # The deduction path: a calm second confrontation breaks her without warmth.
    fsm9 = SuspectFSM()
    drop(fsm9, "wound")
    drop(fsm9, "corridor")
    fsm9.transition(confront("wound"))        # -> Guilty, 1 confronted
    fsm9.transition(confront("corridor"))     # calm deduction, 2 confronted
    results.append(check("calm second confrontation -> Remorseful", fsm9.get_state() is State.REMORSEFUL))

    # ---- De-escalation within the not-aware band ---------------------------
    fsm10 = SuspectFSM()
    fsm10.transition(probe())       # Calm -> Defensive
    fsm10.transition(reassure())    # Defensive -> Calm
    results.append(check("reassurance walks Defensive -> Calm", fsm10.get_state() is State.CALM))

    # ---- reset clears the nugget economy ------------------------------------
    fsm9.reset()
    results.append(check(
        "reset clears nuggets",
        not fsm9.nuggets_dropped and not fsm9.nuggets_confronted
        and fsm9.pending_drop is None and fsm9.last_confront is None,
    ))

    # ---- Keyword fallback produces a usable Signal -------------------------
    sig = classify_keywords("your fingerprints are on the weapon")
    results.append(check("keyword fallback reads evidence", sig.evidence == "strong"))
    sig2 = classify_keywords("what is your alibi for that evening")
    results.append(check("keyword fallback reads probing", sig2.probing is True))
    sig3 = classify_keywords("How was he killed, exactly?")
    results.append(check("keyword fallback reads topic", sig3.topic == "wound"))
    sig4 = classify_keywords(
        "You said you never went down the east corridor, yet you saw him in the study doorway."
    )
    results.append(check("keyword fallback reads confrontation", sig4.nugget == "corridor"))
    sig5 = classify_keywords("Was he stabbed in the neck?")
    results.append(check(
        "keyword fallback: topic mention is not a confrontation",
        sig5.nugget == "none",
    ))

    # ---- Bookkeeping --------------------------------------------------------
    results.append(check("transitions are logged", len(fsm10.history) == 2))
    results.append(check(
        "nugget events are logged",
        fsm8.history[-1]["confronted"] == ["corridor", "wound"],
    ))

    # Verdict scoring matches ground truth.
    accusation_correct = ("guilty" == "guilty") == SUSPECT_IS_GUILTY
    results.append(check("guilty verdict scores correct", accusation_correct))

    # ---- History trimming (llm_client.trim_history) -------------------------
    def convo(pairs):
        msgs = []
        for i in range(pairs):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append({"role": "assistant", "content": f"answer {i}"})
        msgs.append({"role": "user", "content": "current question"})
        return msgs

    short = convo(2)
    results.append(check("trim is a no-op below the cap", trim_history(short, 6) == short))
    long = convo(10)
    trimmed = trim_history(long, 6)
    results.append(check("trim caps at N pairs + current question", len(trimmed) == 13))
    results.append(check("trim keeps the newest turns", trimmed[-1] == long[-1] and trimmed[0] in long))
    results.append(check(
        "trim never returns assistant-first",
        all(trim_history(convo(p), 3)[0]["role"] == "user" for p in range(1, 8)),
    ))
    results.append(check("trim 0 disables", trim_history(long, 0) == long))

    # ---- Established facts survive trimming ----------------------------------
    # Nothing has happened yet: the recap must stay out of the prompt entirely.
    fsm11 = SuspectFSM()
    results.append(check("fresh FSM has no established facts", fsm11.get_established_facts() == ""))
    results.append(check(
        "no facts header in a fresh prompt",
        "Established earlier" not in fsm11.get_system_prompt(),
    ))
    # After a slip, the recap pins it even if the turn scrolls out of the window.
    drop(fsm11, "wound")
    facts = fsm11.get_established_facts()
    results.append(check("dropped slip enters the facts", "neck" in facts))
    results.append(check("facts enter the system prompt", "never deny having said it" in fsm11.get_system_prompt()))
    # The recap only describes what was said: it must not leak guilt into the
    # not-aware band (the same gating the overlays are built around).
    results.append(check("facts do not leak guilt when not aware", fsm11.aware is False
                         and "responsible" not in fsm11.get_system_prompt().lower()))
    # A landed confrontation is recorded too.
    fsm11.transition(confront("wound"))
    results.append(check(
        "confrontation enters the facts",
        "already confronted you" in fsm11.get_established_facts(),
    ))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
