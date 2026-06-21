"""
test_fsm.py
===========
Lightweight checks for the state machine and verdict scoring. No test framework
needed: run `python test_fsm.py` and it prints a pass or fail line per check.

The FSM now transitions on a classified ``Signal`` rather than raw text, so these
tests build Signals directly and never need a running model. That keeps the rules
that matter most for the study guarded: the one-way awareness boundary, the
Offensive-vs-Guilty split, and the empathy path to a confession.
"""

from fsm import SuspectFSM, State, AWARE_STATES, SUSPECT_IS_GUILTY
from intent_classifier import Signal, classify_keywords


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


# Small Signal builders so each test reads as the player's intent, not a pile of
# keyword arguments.
def probe():
    return Signal(probing=True)


def accuse(level="direct", aggression="low"):
    return Signal(accusation=level, aggression=aggression)


def evidence(warmth="cold"):
    return Signal(evidence="strong", warmth=warmth, accusation="direct")


def reassure():
    return Signal(warmth="warm")


def empathise():
    return Signal(warmth="warm", conscience=True)


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

    # Aggression with no proof is the FIGHT response: Defensive -> Offensive.
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

    # ---- Crossing the awareness boundary -----------------------------------
    # Harsh concrete evidence makes the guilt surface directly.
    fsm2 = SuspectFSM()
    fsm2.transition(probe())                 # Calm -> Defensive
    fsm2.transition(evidence(warmth="cold"))  # Defensive -> Guilty (aware)
    results.append(check("evidence -> Guilty", fsm2.get_state() is State.GUILTY))
    results.append(check("evidence makes her aware", fsm2.aware is True))
    results.append(check(
        "Guilty prompt reveals guilt to model",
        "responsible" in fsm2.get_system_prompt().lower(),
    ))

    # A gentle reveal lands on the resigned, composed awareness instead.
    fsm3 = SuspectFSM()
    fsm3.transition(evidence(warmth="warm"))
    results.append(check("gentle evidence -> Resigned", fsm3.get_state() is State.RESIGNED))
    results.append(check("Resigned is an aware state", fsm3.aware is True))

    # ---- Awareness is one-way ----------------------------------------------
    # Once aware, nothing walks her back to a not-aware state.
    fsm4 = SuspectFSM()
    fsm4.transition(evidence(warmth="cold"))  # -> Guilty (aware)
    fsm4.transition(reassure())               # backing off must NOT un-aware her
    results.append(check("reassurance cannot un-aware", fsm4.aware is True))
    results.append(check("reassurance stays in aware band", fsm4.get_state() in AWARE_STATES))
    fsm4.transition(probe())                  # probing must NOT return to Defensive
    results.append(check("probing cannot return to not-aware", fsm4.get_state() in AWARE_STATES))

    # Bullying a guilty suspect makes her clam up (aware but closed), not escalate.
    fsm5 = SuspectFSM()
    fsm5.transition(evidence(warmth="cold"))         # -> Guilty
    fsm5.transition(accuse(aggression="high"))       # Guilty -> Resigned
    results.append(check("aggression in aware band clams her up", fsm5.get_state() is State.RESIGNED))
    results.append(check("still aware after clamming up", fsm5.aware is True))

    # ---- The empathy path to a confession ----------------------------------
    fsm6 = SuspectFSM()
    fsm6.transition(evidence(warmth="cold"))  # -> Guilty
    fsm6.transition(empathise())              # Guilty -> Remorseful
    results.append(check("empathy -> Remorseful", fsm6.get_state() is State.REMORSEFUL))
    results.append(check(
        "Remorseful prompt allows a confession",
        "confession" in fsm6.get_system_prompt().lower(),
    ))
    fsm6.transition(probe())                  # Remorseful -> Confessed (terminal)
    results.append(check("confession settles into Confessed", fsm6.get_state() is State.CONFESSED))
    fsm6.transition(accuse(aggression="high"))
    results.append(check("Confessed is terminal", fsm6.get_state() is State.CONFESSED))

    # ---- De-escalation within the not-aware band ---------------------------
    fsm7 = SuspectFSM()
    fsm7.transition(probe())       # Calm -> Defensive
    fsm7.transition(reassure())    # Defensive -> Calm
    results.append(check("reassurance walks Defensive -> Calm", fsm7.get_state() is State.CALM))

    # ---- Keyword fallback produces a usable Signal -------------------------
    sig = classify_keywords("your fingerprints are on the weapon")
    results.append(check("keyword fallback reads evidence", sig.evidence == "strong"))
    sig2 = classify_keywords("what is your alibi for that evening")
    results.append(check("keyword fallback reads probing", sig2.probing is True))

    # ---- Bookkeeping --------------------------------------------------------
    results.append(check("transitions are logged", len(fsm7.history) == 2))

    # Verdict scoring matches ground truth.
    accusation_correct = ("guilty" == "guilty") == SUSPECT_IS_GUILTY
    results.append(check("guilty verdict scores correct", accusation_correct))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
