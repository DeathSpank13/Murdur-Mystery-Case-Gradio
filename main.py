"""
main.py
=======
Entry point for the NPC interrogation prototype.

Before launching the UI it checks whether the local llama.cpp server is up and
prints a hint if it is not, so you are not left wondering why Dynamic mode
returns a fallback line. The Static mode works regardless of the server.

Run:
    python main.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import threading

import intent_classifier
import llm_client
import static_dialogue
from fsm import SuspectFSM
from ui import build_app, BRANCHING_CSS


def _warm_up_model():
    """Prime both server slots so the first real turn is not the slow one.

    The first classifier call after a server start pays the full few-shot
    prefix evaluation (plus lazy backend shader compilation), which showed up
    as a 60s+ first turn in benchmarks. One dummy classifier call caches that
    prefix in one slot; a one-token in-character call caches the Calm persona
    in the other. Runs in a daemon thread so the UI comes up immediately --
    the player's first message typically lands well after this finishes.
    """
    intent_classifier.classify("Good evening.", None)
    llm_client.get_response(
        SuspectFSM().get_system_prompt(),
        [{"role": "user", "content": "Good evening."}],
        max_tokens=1,
    )
    print("Model warm-up complete. First dynamic turn will be full speed.")


def main():
    if llm_client.server_is_up():
        print("Local model server detected on port 8080. Dynamic mode is ready.")
        print("Warming up the model in the background...")
        threading.Thread(target=_warm_up_model, daemon=True).start()
    else:
        print(
            "Note: local model server not detected on port 8080.\n"
            "Static mode will still work. To enable Dynamic mode, start the "
            "server in another terminal, for example (CUDA build, see README "
            "'Performance tuning' for the download):\n"
            "    C:\\llama-cuda\\llama-server.exe "
            "-hf bartowski/Wayfarer-12B-GGUF:Q4_K_M "
            "-c 8192 -np 2 -ngl 34 -fa on -ctk q8_0 -ctv q8_0\n"
            "(flags benchmarked for an 8GB NVIDIA GPU; winget's Vulkan build "
            "needs different -ngl, see README; benchmark_llm.py re-tunes for "
            "other hardware)\n"
        )

    # Build the Static-mode embedding model and retrieval index up front (it
    # downloads the model on first ever run) so the first study turn isn't slow.
    print("Preparing Static mode (loading embedding model and building index)...")
    static_dialogue.warm_up()
    print("Static mode ready.")

    app = build_app()
    # share=False keeps the demo local. Set share=True only if you need a
    # temporary public link for a remote test session. css dims the branching
    # tab's "already chosen" buttons (Gradio 6 reads custom css here, not on
    # the Blocks() constructor).
    app.launch(share=False, css=BRANCHING_CSS)


if __name__ == "__main__":
    main()
