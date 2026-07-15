"""
sync_qa_data.py
===============
Regenerate docs/js/static_qa_data.js from data/suspect_qa.json.

The browser demo (docs/) mirrors the Python static condition but cannot read
the JSON directly when served as plain files, so the database is shipped as an
ES module. This script is the only way that file should be written: run it
whenever data/suspect_qa.json changes and commit both files together.

    python sync_qa_data.py
"""

import json
import os

REPO_ROOT = os.path.dirname(__file__)
SOURCE = os.path.join(REPO_ROOT, "data", "suspect_qa.json")
TARGET = os.path.join(REPO_ROOT, "docs", "js", "static_qa_data.js")

HEADER = """\
// static_qa_data.js
// =================
// GENERATED FILE -- do not edit by hand. Regenerate with `python
// sync_qa_data.py` whenever data/suspect_qa.json (the canonical database)
// changes. Each entry pairs an ordered list of pre-written response variants
// (plus optional repeat_responses spoken once the variants run out) with
// several natural-language example questions that should map to it, and a
// topic_hint spoken by the clarify band. static_dialogue.js embeds the
// questions once and routes the player's input by meaning.

export const SUSPECT_QA = """


def main():
    with open(SOURCE, "r", encoding="utf-8") as f:
        data = json.load(f)
    body = json.dumps(data, indent=2, ensure_ascii=False)
    with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + body + ";\n")
    print(f"Wrote {len(data)} entries to {os.path.relpath(TARGET, REPO_ROOT)}")


if __name__ == "__main__":
    main()
