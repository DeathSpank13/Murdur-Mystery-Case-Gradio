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

import llm_client
import static_dialogue
from ui import build_app, BRANCHING_CSS


def main():
    if llm_client.server_is_up():
        print("Local model server detected on port 8080. Dynamic mode is ready.")
    else:
        print(
            "Note: local model server not detected on port 8080.\n"
            "Static mode will still work. To enable Dynamic mode, start the "
            "server in another terminal, for example:\n"
            "    llama-server -hf bartowski/Wayfarer-12B-GGUF:Q4_K_M "
            "-c 8192 -np 2 -ngl 28 -fa on -ctk q8_0 -ctv q8_0\n"
            "(flags benchmarked for an 8GB GPU; see README 'Performance "
            "tuning' and benchmark_llm.py to re-tune for other hardware)\n"
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
