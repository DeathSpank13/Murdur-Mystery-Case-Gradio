"""
test_fsm.py
===========
Lightweight checks for the state machine and verdict scoring. No test framework
needed: run `python test_fsm.py` and it prints a pass or fail line per check.
These guard the behaviours that matter most for the study, so a later tweak to
the trigger lists cannot silently break escalation.
"""

from fsm import SuspectFSM, State, SUSPECT_IS_GUILTY


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def run():
    results = []

    # Starts Calm.
    fsm = SuspectFSM()
    results.append(check("starts in Calm", fsm.get_state() is State.CALM))

    # Calm prompt does not reveal guilt.
    results.append(check(
        "Calm prompt hides guilt",
        "responsible" not in fsm.get_system_prompt().lower(),
    ))

    # Probing escalates Calm to Suspicious.
    fsm.transition("what is your alibi")
    results.append(check("probing -> Suspicious", fsm.get_state() is State.SUSPICIOUS))

    # Suspicious prompt now carries the secret.
    results.append(check(
        "Suspicious prompt reveals guilt to model",
        "responsible" in fsm.get_system_prompt().lower(),
    ))

    # Accusation escalates Suspicious to Defensive.
    fsm.transition("you killed him, confess")
    results.append(check("accusation -> Defensive", fsm.get_state() is State.DEFENSIVE))

    # De-escalation walks back one step.
    fsm.transition("i'm sorry, no offense")
    results.append(check("apology -> Suspicious", fsm.get_state() is State.SUSPICIOUS))

    # A single accusation from Calm only reaches Suspicious, not Defensive.
    fsm2 = SuspectFSM()
    fsm2.transition("you killed him")
    results.append(check("one accusation from Calm -> Suspicious", fsm2.get_state() is State.SUSPICIOUS))

    # History is recorded.
    results.append(check("transitions are logged", len(fsm.history) == 3))

    # Verdict scoring matches ground truth.
    accusation_correct = ("guilty" == "guilty") == SUSPECT_IS_GUILTY
    results.append(check("guilty verdict scores correct", accusation_correct))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
