"""
test_dialogue.py
================
Lightweight checks for the branching dialogue engine. No test framework needed:
run `python test_dialogue.py` and it prints a pass or fail line per check.

These guard the three consequence rules that are the whole point of the mode, so
a later edit to DIALOGUE_TREE or the engine cannot silently break them:

    once             a one-time question disappears after it is asked
    exclusive_group  choosing one fork hides the siblings for the rest of the run
    goto / Back      nesting descends and climbs back, and consumption persists
                     across navigation
"""

from branching_dialogue import DialogueEngine, BACK_ID


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def ids(engine):
    """The set of option ids currently visible at the engine's node."""
    return {opt["id"] for opt in engine.available_options()}


def run():
    results = []

    # Seeds the transcript with the start node's intro line.
    engine = DialogueEngine()
    results.append(check(
        "intro line seeds the transcript",
        engine.transcript and engine.transcript[0][0] == "npc",
    ))

    # Root offers all five top-level topics, no Back at the root.
    results.append(check(
        "root shows topics and no Back",
        {"party", "alibi", "relationship", "weapon", "press", "reassure"} == ids(engine)
        and BACK_ID not in ids(engine),
    ))

    # ONE-TIME: the top-level alibi question is `once`. Visiting it descends
    # (goto), so come back up and confirm it is gone.
    engine.choose("alibi")
    engine.choose(BACK_ID)
    results.append(check(
        "one-time question is consumed after use",
        "alibi" not in ids(engine),
    ))

    # EXCLUSIVE: choosing one approach locks the whole group, hiding the sibling.
    results.append(check("both approaches visible before choosing", {"press", "reassure"} <= ids(engine)))
    engine.choose("press")          # descends into the "press" node
    engine.choose(BACK_ID)          # climb back to root
    results.append(check(
        "exclusive sibling hidden after committing to one",
        "press" not in ids(engine) and "reassure" not in ids(engine),
    ))

    # NESTING + Back: descending shows a child node with its own options and a
    # Back option; Back returns to the parent.
    engine.choose("relationship")   # goto -> relationship node
    results.append(check("descending shows child options", "business" in ids(engine)))
    results.append(check("child node offers Back", BACK_ID in ids(engine)))
    engine.choose("business")       # deeper still
    results.append(check("second level of nesting", "debt" in ids(engine)))
    engine.choose(BACK_ID)          # back to relationship
    results.append(check("Back climbs one level", "business" in ids(engine)))

    # PERSISTENCE: a one-time question used deep in the tree stays consumed after
    # navigating away and back.
    engine.choose("disagreement")   # `once` option on the relationship node
    engine.choose(BACK_ID)          # to root
    engine.choose("relationship")   # back down
    results.append(check(
        "consumption persists across navigation",
        "disagreement" not in ids(engine),
    ))

    # A repeatable pick is remembered in `chosen` (so the UI can dim it) yet
    # stays on offer, unlike a one-time question. Use a fresh engine so this is
    # independent of the navigation above.
    fresh = DialogueEngine()
    fresh.choose("weapon")          # goto topic, repeatable
    fresh.choose("who_handled")     # repeatable leaf on the weapon node
    results.append(check(
        "repeatable pick is recorded in chosen",
        "who_handled" in fresh.chosen and "weapon" in fresh.chosen,
    ))
    results.append(check(
        "repeatable pick stays available after use",
        "who_handled" in ids(fresh),
    ))

    # Choosing an unavailable id is a loud error, not a silent no-op.
    raised = False
    try:
        engine.choose("disagreement")
    except KeyError:
        raised = True
    results.append(check("stale/unavailable id raises", raised))

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    run()
