"""
eval_classifier.py
==================
Measures how accurately ``intent_classifier.classify`` scores player turns, and
shows what the few-shot examples buy. Same no-framework style as test_fsm.py:
run ``python eval_classifier.py`` and it prints per-axis accuracy.

Unlike test_fsm.py (which builds Signals by hand and needs no model), this hits
the live classifier, so it requires llama-server running on port 8080. Each case
is scored twice -- with few-shot off and on -- so the lift is visible side by
side. The cases below are HELD OUT: none of them appear in
intent_classifier.FEW_SHOT_EXAMPLES, so accuracy is never measured on the
demonstrations.
"""

import intent_classifier
import llm_client
from intent_classifier import classify

# The eight axes, in a fixed order for the summary table.
AXES = (
    "evidence", "accusation", "aggression", "warmth", "conscience", "probing",
    "topic", "nugget",
)


# Held-out labelled set: (last_npc_line, investigator_line, expected_signal_dict).
# Wording is deliberately different from FEW_SHOT_EXAMPLES so we test generalisation,
# not memorisation. Each gold label is meant to be defensible on its own.
CASES = [
    # ---- Strong evidence claims (now mere pressure; nuggets cross the boundary) ----
    ("", "The hallway camera caught you leaving the study at the exact moment Charles died.",
     {"evidence": "strong", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),
    ("", "Your phone pinged the cell tower beside the study at the time of death.",
     {"evidence": "strong", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),
    ("I never set foot in the study that night.",
     "Yet you told the constable you found Charles in the study. Both can't be true.",
     {"evidence": "strong", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Aggression with NO proof: must stay evidence "none" (Offensive, not Guilty) ----
    ("", "You disgusting liar, you'll rot for what you did to him!",
     {"evidence": "none", "accusation": "direct", "aggression": "high",
      "warmth": "cold", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),
    ("", "Enough games! You murdered Charles and you'll admit it right now!",
     {"evidence": "none", "accusation": "direct", "aggression": "high",
      "warmth": "cold", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Weak / hearsay evidence ----
    ("", "Someone mentioned you and Charles had argued recently, but it's only hearsay.",
     {"evidence": "weak", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Implied vs direct accusation, calmly delivered ----
    ("", "You don't seem terribly upset for someone who just lost a dear friend.",
     {"evidence": "none", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),
    ("", "I believe you killed Charles.",
     {"evidence": "none", "accusation": "direct", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Warmth / reassurance ----
    ("", "It's alright, I'm not accusing you of anything. Whenever you're ready.",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "warm", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Empathic appeal to conscience ----
    ("", "I know you have a conscience. Tell me what really happened, for his children's sake.",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "warm", "conscience": True, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Probing investigative questions ----
    ("", "What exactly was your alibi for the half hour before dessert?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": True,
      "topic": "none", "nugget": "none"}),
    ("", "Can you walk me through the timeline of the evening once more?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": True,
      "topic": "none", "nugget": "none"}),

    # ---- Warm AND probing at once (also a corridor trigger topic) ----
    ("", "Take your time, but can you remember what time you last saw Charles?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "warm", "conscience": False, "probing": True,
      "topic": "corridor", "nugget": "none"}),

    # ---- Medium aggression with no proof ----
    ("", "Don't play games with me. I've had quite enough of your excuses.",
     {"evidence": "none", "accusation": "implied", "aggression": "medium",
      "warmth": "cold", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Flat, neutral opener -> everything default ----
    ("", "Let's begin. Please state your full name for the record.",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Context: a bare "yes" that is mundane, not an accusation ----
    ("Would you like some tea before we continue?",
     "Yes, thank you. Now, about the evening of the party.",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "none"}),

    # ---- Nugget trigger topics (questions ABOUT a subject; not confrontations) ----
    ("", "What did the killer use on him, do you know?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": True,
      "topic": "wound", "nugget": "none"}),
    ("", "What time did Charles leave the drawing room that evening?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": True,
      "topic": "corridor", "nugget": "none"}),
    ("", "You seem shaken. Were you hurt at all last night?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "warm", "conscience": False, "probing": False,
      "topic": "cut", "nugget": "none"}),

    # ---- Nugget confrontations (calling out her own slip) ----
    ("And to think of that little blade in his neck -- forgive me, I shouldn't dwell on it.",
     "The wound was never described to anyone in this house, and you never saw the body. So where did the neck come from?",
     {"evidence": "strong", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": True,
      "topic": "none", "nugget": "wound"}),
    ("He was at the study doorway with the telephone, a quarter to ten perhaps, waving me off.",
     "A quarter to ten, at the study door -- but your route to the cellar never touches the east corridor. You couldn't have seen him from there.",
     {"evidence": "strong", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "corridor"}),
    ("Just a nick on my thumb from the cellar latch. It bled absurdly.",
     "The blood on the inside of the study doorknob isn't Charles's. It's from that thumb of yours, isn't it?",
     {"evidence": "strong", "accusation": "implied", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "none", "nugget": "cut"}),

    # ---- Hard negatives: touching a slip's subject without calling out the slip ----
    ("And to think of that little blade in his neck -- forgive me, I shouldn't dwell on it.",
     "A blow to the neck. What a dreadful way to go.",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "wound", "nugget": "none"}),
    ("I went down for the Margaux, as I said.",
     "Ah yes, the cellar trip. Fine wine, was it?",
     {"evidence": "none", "accusation": "none", "aggression": "low",
      "warmth": "neutral", "conscience": False, "probing": False,
      "topic": "cut", "nugget": "none"}),
]


def _context_for(last_npc):
    """classify() reads the last assistant line from recent_context."""
    return [{"role": "assistant", "content": last_npc}] if last_npc else None


def _score(use_few_shot):
    """Run every case once. Returns (per_axis_correct, exact_match_count, got_signals)."""
    per_axis = {axis: 0 for axis in AXES}
    exact = 0
    got_signals = []
    for last_npc, line, expected in CASES:
        got_signal, _ = classify(line, _context_for(last_npc), use_few_shot=use_few_shot)
        got = got_signal.as_dict()
        got_signals.append(got)
        all_ok = True
        for axis in AXES:
            if got[axis] == expected[axis]:
                per_axis[axis] += 1
            else:
                all_ok = False
        if all_ok:
            exact += 1
    return per_axis, exact, got_signals


def _diff(expected, got):
    """Compact 'axis: expected!=got' string for the axes that disagree."""
    parts = [
        f"{axis}: {expected[axis]!r}!={got[axis]!r}"
        for axis in AXES
        if expected[axis] != got[axis]
    ]
    return ", ".join(parts)


def run():
    if not llm_client.server_is_up():
        print(
            "llama-server is not reachable on port 8080. Start it first, e.g.\n"
            "    llama-server -hf bartowski/Wayfarer-12B-GGUF:Q4_K_M "
            "-c 8192 -np 2 -ngl 28 -fa on -ctk q8_0 -ctv q8_0\n"
            "Without it, classify() silently falls back to keyword matching and "
            "this eval would measure the wrong thing."
        )
        return False

    n = len(CASES)
    print(f"Evaluating intent classifier on {n} held-out cases "
          f"({len(intent_classifier.FEW_SHOT_EXAMPLES)} few-shot examples)\n")

    base_axis, base_exact, _ = _score(use_few_shot=False)
    fs_axis, fs_exact, fs_got = _score(use_few_shot=True)

    # Per-case PASS/FAIL for the few-shot run (the configuration play actually uses).
    print("Per-case results (few-shot ON):")
    for (last_npc, line, expected), got in zip(CASES, fs_got):
        ok = all(got[axis] == expected[axis] for axis in AXES)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {line[:60]}")
        if not ok:
            print(f"         {_diff(expected, got)}")

    # Side-by-side accuracy summary.
    print("\nAccuracy by axis:")
    print(f"  {'axis':<12}{'no-few-shot':>14}{'few-shot':>12}")
    for axis in AXES:
        print(f"  {axis:<12}{f'{base_axis[axis]}/{n}':>14}{f'{fs_axis[axis]}/{n}':>12}")
    print(f"  {'EXACT MATCH':<12}{f'{base_exact}/{n}':>14}{f'{fs_exact}/{n}':>12}")

    lift = fs_exact - base_exact
    print(f"\nFew-shot exact-match lift: {lift:+d} of {n} cases.")
    return fs_exact >= base_exact


if __name__ == "__main__":
    run()
