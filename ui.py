"""
ui.py
=====
Gradio interface for the interrogation prototype.

Two suspect implementations sit behind neutral labels (Detective A and
Detective B) so participants are not primed toward "the clever AI one":

    static   Uses static_dialogue.get_response(). A fixed dialogue tree.
    dynamic  Uses SuspectFSM to pick a persona, then llm_client to generate the
             reply with the local model.

Which label maps to which condition is randomised per session and kept only in
the logs. A "Researcher view" toggle (off for participants, on for demos)
reveals the live FSM state and the last response latency.

The per turn logic lives in handle_turn(), kept separate from the UI layout so
it can be exercised without launching Gradio.
"""

import random
import time

import gradio as gr

import static_dialogue
import llm_client
import logger
import intent_classifier
import branching_dialogue
from branching_dialogue import DialogueEngine
from fsm import SuspectFSM, SUSPECT_IS_GUILTY
from nuggets import NUGGETS_FOR_CONFESSION


# Number of option-button slots to pre-create for the branching dialogue tab.
# Gradio needs a fixed set of components, so we make enough for the largest node
# (the root menu) plus one for the synthetic Back option, and hide the unused
# slots each turn.
MAX_OPTIONS = max(
    len(node["options"]) for node in branching_dialogue.DIALOGUE_TREE.values()
) + 1

# Styling for the branching tab: dim the buttons for options the player has
# already chosen. Buttons get the "option-used" class via gr.update; this rule
# fades them. Gradio 6 reads custom css from launch(css=...), so main.py passes
# this there (the Blocks(css=...) constructor argument is deprecated and ignored).
BRANCHING_CSS = ".option-used { opacity: 0.55; }"


INTRO = (
    "## Interrogation: The Whitmore Case\n"
    "You are the investigator. Eleanor Vance hosted last night's dinner party, "
    "where Charles Whitmore was found dead in the study. Question her, then "
    "decide: is she guilty?\n\n"
    "**What the case file tells you:**\n"
    "1. Charles was found dead in the study at 10:15 pm by Daniel Reeve, sent "
    "to fetch him for dessert. Guests were kept out of the room, and details "
    "of the wound have not been released to anyone in the house.\n"
    "2. The coroner puts the time of death between 9:30 and 10:00 pm.\n"
    "3. The study sits at the end of the east corridor; the kitchen, back "
    "stairs and wine cellar are at the west end of the house.\n"
    "4. Forensics found a fresh smear of blood on the inside knob of the study "
    "door -- not Charles's type. The letter opener that killed him was wiped "
    "clean.\n\n"
    "Listen closely: a suspect's own words are worth more than any accusation. "
    "If something she says does not fit, put it to her.\n\n"
    "You can question two detectives' suspects, **Detective A** and "
    "**Detective B**. Interview each as you like, then submit a verdict for "
    "the one you are judging."
)


def new_label_mapping():
    """
    Randomly assign the two conditions to the two neutral labels for a session.
    Returns e.g. {"Detective A": "dynamic", "Detective B": "static"}.
    """
    conditions = ["static", "dynamic"]
    random.shuffle(conditions)
    return {"Detective A": conditions[0], "Detective B": conditions[1]}


def handle_turn(
    player_input, chat, detective_label, fsm, static_counts, llm_history,
    label_mapping, session_logger, researcher_on, simulate_delay=True,
):
    """
    Process one player turn.

    Resolves the chosen neutral label to its hidden condition, generates the
    reply, logs the turn (with latency and FSM state), and returns updated UI
    values. State readout and latency are only made visible when researcher_on.
    """
    chat = chat or []
    condition = label_mapping.get(detective_label, "dynamic")

    def state_update(text):
        return gr.update(value=text, visible=researcher_on)

    if not player_input or not player_input.strip():
        current = fsm.get_state().value if condition == "dynamic" else "n/a (static script)"
        return (
            chat, "",
            state_update(f"**Suspect state:** {current}"),
            gr.update(visible=researcher_on),
            fsm, static_counts, llm_history, label_mapping, session_logger,
        )

    chat.append({"role": "user", "content": player_input})

    signal_dict = None
    nugget_event = None
    if condition == "static":
        start = time.perf_counter()
        reply = static_dialogue.get_response(player_input, static_counts)
        real_latency_ms = (time.perf_counter() - start) * 1000.0
        # Add a length-scaled artificial delay so the instant lookup doesn't
        # give the static condition away (and confound the study). A researcher
        # can disable it via the dev toggle for fast iteration.
        simulated_latency_ms = static_dialogue.simulate_latency(reply) if simulate_delay else 0.0
        latency_ms = real_latency_ms + simulated_latency_ms   # perceived total
        state_value = "n/a (static script)"
    else:
        # Classify the turn on its multi-axis Signal first (using the history so
        # far for context), then let the FSM move on the combination of axes. The
        # classification is a separate model call, so it adds to the turn latency.
        signal = intent_classifier.classify(player_input, llm_history)
        signal_dict = signal.as_dict()
        fsm.transition(signal)
        system_prompt = fsm.get_system_prompt()
        attempted_drop = fsm.pending_drop
        llm_history.append({"role": "user", "content": player_input})
        reply, latency_ms = llm_client.get_response(system_prompt, llm_history)
        llm_history.append({"role": "assistant", "content": reply})
        # Confirm whether the slip planned for this reply was actually said;
        # the nugget only becomes confrontable once it is in the transcript.
        dropped = fsm.commit_reply(reply)
        if fsm.last_confront:
            nugget_event = f"confront:{fsm.last_confront}"
        elif dropped:
            nugget_event = f"drop:{dropped}"
        elif attempted_drop:
            nugget_event = f"drop_failed:{attempted_drop}"
        state_value = fsm.get_state().value
        real_latency_ms = latency_ms          # dynamic latency is already real
        simulated_latency_ms = 0.0

    chat.append({"role": "assistant", "content": reply})
    session_logger.log_turn(
        condition, detective_label, player_input, reply, state_value, latency_ms,
        signal=signal_dict,
        real_latency_ms=real_latency_ms,
        simulated_latency_ms=simulated_latency_ms,
        nuggets_dropped=sorted(fsm.nuggets_dropped) if signal_dict else None,
        nuggets_confronted=sorted(fsm.nuggets_confronted) if signal_dict else None,
        nugget_event=nugget_event,
    )

    # Researcher readout: the state, plus the axes that drove it for the dynamic
    # condition so a demo can show *why* she moved, not just where she landed.
    if signal_dict is None:
        state_text = f"**Suspect state:** {state_value}"
    else:
        axes = ", ".join(f"{key}={value}" for key, value in signal_dict.items())
        aware = " (aware)" if fsm.aware else ""
        dropped_list = ", ".join(sorted(fsm.nuggets_dropped)) or "-"
        confronted_list = ", ".join(sorted(fsm.nuggets_confronted)) or "-"
        nugget_text = (
            f"**Nuggets:** dropped [{dropped_list}] / confronted "
            f"[{confronted_list}] ({NUGGETS_FOR_CONFESSION} needed)"
        )
        if nugget_event:
            nugget_text += f" / {nugget_event}"
        state_text = (
            f"**Suspect state:** {state_value}{aware}  \n**Read:** {axes}  \n"
            + nugget_text
        )

    return (
        chat, "",
        state_update(state_text),
        gr.update(value=f"**Last response latency:** {latency_ms:.0f} ms", visible=researcher_on),
        fsm, static_counts, llm_history, label_mapping, session_logger,
    )


def submit_verdict(verdict_choice, confidence, detective_label, label_mapping, session_logger):
    """Score and log the participant's accusation, then reveal the truth."""
    condition = label_mapping.get(detective_label, "dynamic")
    accusation = "guilty" if "guilty" in (verdict_choice or "").lower() else "innocent"
    correct = (accusation == "guilty") == SUSPECT_IS_GUILTY
    session_logger.log_verdict(condition, detective_label, accusation, correct, int(confidence))

    head = "Your judgement was correct." if correct else "Your judgement was incorrect."
    truth = "Eleanor Vance was, in fact, responsible for Charles's death."
    return f"**{head}** {truth} (Recorded for {detective_label}.)", session_logger


def toggle_researcher(on):
    """Show or hide the researcher only panels (and the dev delay toggle)."""
    return gr.update(visible=on), gr.update(visible=on), gr.update(visible=on)


def reset_session():
    """Clear the conversation and start a fresh, freshly randomised session."""
    return (
        [],                                  # chatbot
        "",                                  # msg
        gr.update(value="**Suspect state:** Calm", visible=False),
        gr.update(value="", visible=False),  # latency
        SuspectFSM(),                        # fresh FSM
        {},                                  # fresh static counts
        [],                                  # fresh llm history
        new_label_mapping(),                 # re-randomise A/B mapping
        logger.SessionLogger(),              # new session log
        "",                                  # clear verdict output
    )


# ---------------------------------------------------------------------------
# Branching dialogue tab (a separate mode; does not touch the blinded study)
# ---------------------------------------------------------------------------
# The DialogueEngine (branching_dialogue.py) owns all the logic; this layer only
# turns its current state into Gradio updates: the transcript becomes chat
# messages, and the available options become button labels, with unused button
# slots hidden.

def branching_outputs(engine):
    """
    Build the full output tuple for any branching event.

    Returns [chat_messages, engine, option_ids, *button_updates], matching the
    `branch_outputs` wiring in build_app(). option_ids maps each visible button
    slot back to the option id the engine expects, so a click knows what was
    chosen.
    """
    role = {"player": "user", "npc": "assistant"}
    chat = [{"role": role[speaker], "content": text} for speaker, text in engine.transcript]

    options = engine.available_options()
    option_ids = [opt["id"] for opt in options]

    button_updates = []
    for i in range(MAX_OPTIONS):
        if i < len(options):
            # Dim options the player has already picked (repeatable questions and
            # visited topics) so they can see where they've been, game-style.
            used = options[i]["id"] in engine.chosen
            button_updates.append(gr.update(
                value=options[i]["text"],
                visible=True,
                elem_classes=["option-used"] if used else [],
            ))
        else:
            button_updates.append(gr.update(value="", visible=False, elem_classes=[]))

    return [chat, engine, option_ids, *button_updates]


def load_branching(engine):
    """Render the opening menu for the per-session engine when the page loads."""
    return branching_outputs(engine or DialogueEngine())


def on_branching_choice(index, engine, option_ids):
    """Apply the option behind the clicked button slot, then re-render."""
    engine = engine or DialogueEngine()
    if option_ids and index < len(option_ids):
        engine.choose(option_ids[index])
    return branching_outputs(engine)


def reset_branching():
    """Start the branching conversation over from the root node."""
    return branching_outputs(DialogueEngine())


def build_app():
    """Construct and return the Gradio Blocks app."""
    with gr.Blocks(title="NPC Interrogation Prototype") as app:

        # Per session state. Passing a callable makes Gradio build a fresh
        # object for each browser session, so concurrent testers stay isolated.
        fsm_state = gr.State(SuspectFSM)
        static_counts_state = gr.State(dict)
        llm_history_state = gr.State(list)
        label_mapping_state = gr.State(new_label_mapping)
        logger_state = gr.State(logger.SessionLogger)

        # Branching dialogue per session state.
        branch_engine_state = gr.State(DialogueEngine)
        branch_ids_state = gr.State(list)  # maps button slot -> option id

        with gr.Tabs():
            # ---- Tab 1: the blinded A/B study, exactly as before -------------
            with gr.Tab("Interrogation (study)"):
                gr.Markdown(INTRO)

                with gr.Row():
                    detective = gr.Radio(
                        choices=["Detective A", "Detective B"],
                        value="Detective A",
                        label="Suspect to question",
                    )
                    researcher_view = gr.Checkbox(
                        value=False,
                        label="Researcher view (show FSM state and latency)",
                    )
                    simulate_delay_toggle = gr.Checkbox(
                        value=True,                      # delay on by default
                        label="Simulate static response delay (dev)",
                        visible=False,                   # only shown in researcher view
                    )

                with gr.Row():
                    state_label = gr.Markdown("**Suspect state:** Calm", visible=False)
                    latency_label = gr.Markdown("", visible=False)

                chatbot = gr.Chatbot(label="Interrogation", height=420)
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Where were you when Charles died?",
                )
                with gr.Row():
                    send_btn = gr.Button("Ask", variant="primary")
                    reset_btn = gr.Button("New session / reset")

                with gr.Group():
                    gr.Markdown("### Make your accusation")
                    verdict_choice = gr.Radio(
                        choices=["She is guilty", "She is innocent"],
                        label="Your verdict",
                    )
                    confidence = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="How confident are you? (1 = guessing, 5 = certain)",
                    )
                    verdict_btn = gr.Button("Submit verdict")
                    verdict_output = gr.Markdown("")

            # ---- Tab 2: the standalone choice-based dialogue tree ------------
            with gr.Tab("Branching dialogue"):
                gr.Markdown(
                    "## Branching dialogue\n"
                    "A choice-based interrogation of Eleanor Vance. Pick a line "
                    "to say; some questions can only be asked **once**, and some "
                    "choices are a **fork** — commit to one and the alternatives "
                    "vanish for this run. Topics open follow-up questions; use "
                    "**← Back** to step out."
                )
                branch_chatbot = gr.Chatbot(
                    label="Interrogation (branching)", height=420
                )
                gr.Markdown("**Choose your next line:**")
                # A fixed pool of buttons; their labels and visibility are set
                # each turn from the engine's available options.
                option_buttons = [
                    gr.Button("", visible=False) for _ in range(MAX_OPTIONS)
                ]
                branch_reset_btn = gr.Button("Restart conversation")

        # Wire study events.
        turn_inputs = [
            msg, chatbot, detective, fsm_state, static_counts_state,
            llm_history_state, label_mapping_state, logger_state, researcher_view,
            simulate_delay_toggle,
        ]
        turn_outputs = [
            chatbot, msg, state_label, latency_label, fsm_state,
            static_counts_state, llm_history_state, label_mapping_state, logger_state,
        ]
        send_btn.click(handle_turn, inputs=turn_inputs, outputs=turn_outputs)
        msg.submit(handle_turn, inputs=turn_inputs, outputs=turn_outputs)

        researcher_view.change(
            toggle_researcher, inputs=researcher_view,
            outputs=[state_label, latency_label, simulate_delay_toggle],
        )

        verdict_btn.click(
            submit_verdict,
            inputs=[verdict_choice, confidence, detective, label_mapping_state, logger_state],
            outputs=[verdict_output, logger_state],
        )

        reset_btn.click(
            reset_session,
            outputs=[
                chatbot, msg, state_label, latency_label, fsm_state,
                static_counts_state, llm_history_state, label_mapping_state,
                logger_state, verdict_output,
            ],
        )

        # Wire branching events. Each button passes its fixed slot index; the
        # default-arg closure captures i so every button gets its own index.
        branch_outputs = [branch_chatbot, branch_engine_state, branch_ids_state, *option_buttons]
        for i, btn in enumerate(option_buttons):
            btn.click(
                lambda engine, ids, idx=i: on_branching_choice(idx, engine, ids),
                inputs=[branch_engine_state, branch_ids_state],
                outputs=branch_outputs,
            )
        branch_reset_btn.click(reset_branching, outputs=branch_outputs)
        # Render the opening menu when the page loads.
        app.load(load_branching, inputs=[branch_engine_state], outputs=branch_outputs)

    return app
