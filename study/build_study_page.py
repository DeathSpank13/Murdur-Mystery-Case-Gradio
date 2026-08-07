"""
build_study_page.py
===================
Regenerate docs/study/index.html from study/questionnaire.md.

The questionnaire is published on GitHub Pages so a reviewer can read the whole
instrument without a Google account, while the actual responses are collected
in three Google Forms. That means the item text would otherwise live in three
places (the markdown, the Apps Script, and the page) and silently drift. This
script removes one of them: the page is generated, never hand-edited.

Run it whenever study/questionnaire.md or study/form_links.json changes, and
commit both files together:

    python study/build_study_page.py

Markdown contract (the parser is deliberately strict):
  * `## Part N - <title>` starts a part card; any other `##` heading is
    researcher-only material and lands in the collapsed block at the bottom.
  * `> **When:** ...` inside a part becomes the card's subtitle.
  * `- **ID** text - [Type] `choice` / `choice` *(note)*` is one question.
    The type annotation in square brackets is what makes a bullet a question.
"""

import html
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO_ROOT, "study", "questionnaire.md")
LINKS = os.path.join(REPO_ROOT, "study", "form_links.json")
TARGET = os.path.join(REPO_ROOT, "docs", "study", "index.html")

REPO_URL = "https://github.com/DeathSpank13/Murdur-Mystery-Case-Gradio"

TITLE_RE = re.compile(r"^# (.+)$")
H2_RE = re.compile(r"^## (.+)$")
H3_RE = re.compile(r"^### (.+)$")
PART_RE = re.compile(r"^Part (\d+) — (.+)$")
ITEM_RE = re.compile(r"^- \*\*([^*]+)\*\* (.+)$")
# Greedy first group so the LAST " — [Type]" wins: item text may itself
# contain an em dash (e.g. "What gave it away — or why couldn't you tell?").
TYPE_RE = re.compile(r"^(.*)\s+—\s+\[([^\]]+)\]\s*(.*)$")
NOTE_RE = re.compile(r"\*\((.+)\)\*")
WHEN_RE = re.compile(r"^\*\*When:\*\*\s*(.+)$")


# --------------------------------------------------------------- parsing ---
def inline(text):
    """Escape HTML, then apply the handful of inline marks the source uses."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", out)
    return out


def parse_item(qid, rest):
    match = TYPE_RE.match(rest)
    if not match:
        return None  # a plain bullet, not a question
    text, qtype, tail = match.group(1), match.group(2), match.group(3)
    note = ""
    note_match = NOTE_RE.search(tail)
    if note_match:
        note = note_match.group(1)
        tail = tail[: note_match.start()] + tail[note_match.end():]
    return {
        "id": qid.strip(),
        "text": text.strip(),
        "type": qtype.strip(),
        "choices": re.findall(r"`([^`]+)`", tail),
        "note": note.strip(),
    }


def split_sections(lines):
    """-> (title, front-matter lines, [(heading, lines), ...])"""
    title, front, sections, current = "", [], [], None
    for line in lines:
        title_match = TITLE_RE.match(line)
        if title_match and not title:
            title = title_match.group(1)
            continue
        heading = H2_RE.match(line)
        if heading:
            current = (heading.group(1), [])
            sections.append(current)
        elif current is None:
            front.append(line)
        else:
            current[1].append(line)
    return title, front, sections


def parse_part(lines):
    """-> list of ('when'|'quote'|'group'|'para'|'item', payload) blocks."""
    blocks, para, quote = [], [], []

    def flush_para():
        if para:
            blocks.append(("para", " ".join(para).strip()))
            del para[:]

    def flush_quote():
        if quote:
            text = " ".join(quote).strip()
            when = WHEN_RE.match(text)
            blocks.append(("when", when.group(1)) if when else ("quote", text))
            del quote[:]

    for raw in lines:
        line = raw.rstrip()
        if line.startswith(">"):
            flush_para()
            quote.append(line.lstrip("> ").strip())
            continue
        flush_quote()
        if not line.strip():
            flush_para()
            continue
        group = H3_RE.match(line)
        if group:
            flush_para()
            blocks.append(("group", group.group(1)))
            continue
        bullet = ITEM_RE.match(line)
        if bullet:
            item = parse_item(bullet.group(1), bullet.group(2))
            if item:
                flush_para()
                blocks.append(("item", item))
                continue
        para.append(line.strip())
    flush_para()
    flush_quote()
    return blocks


def render_prose(lines):
    """Loose renderer for the researcher-only text: paragraphs and lists."""
    out, para, items, kind = [], [], [], None

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            del para[:]

    def flush_list():
        if items:
            tag = "ol" if kind == "ol" else "ul"
            out.append("<%s>" % tag)
            out.extend("  <li>" + inline(i) + "</li>" for i in items)
            out.append("</%s>" % tag)
            del items[:]

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped == "---":
            flush_para()
            flush_list()
            kind = None
            continue
        if line[:1].isspace() and items:  # wrapped continuation of a list item
            items[-1] += " " + stripped
            continue
        heading = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if heading:
            flush_para()
            flush_list()
            kind = None
            level = len(heading.group(1))
            out.append("<h%d>%s</h%d>"
                       % (level, inline(heading.group(2)), level))
            continue
        ordered = re.match(r"^\d+\.\s+(.*)$", stripped)
        unordered = re.match(r"^[-*]\s+(.*)$", stripped)
        if ordered or unordered:
            flush_para()
            want = "ol" if ordered else "ul"
            if kind != want:
                flush_list()
                kind = want
            items.append((ordered or unordered).group(1))
            continue
        flush_list()
        kind = None
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ").strip()
        para.append(stripped)
    flush_para()
    flush_list()
    return out


# -------------------------------------------------------------- rendering ---
def render_item(item):
    bits = ['<span class="qid">%s</span>' % html.escape(item["id"])]
    bits.append(inline(item["text"]))
    if item["type"]:
        bits.append('<span class="badge">%s</span>' % html.escape(item["type"]))
    if item["choices"]:
        choices = " / ".join("<code>%s</code>" % html.escape(c)
                             for c in item["choices"])
        bits.append('<span class="choices">%s</span>' % choices)
    line = "      <li>" + " ".join(bits)
    if item["note"]:
        line += '\n        <span class="qnote">%s</span>' % inline(item["note"])
    return line + "</li>"


def render_card(number, heading, blocks, link):
    items = [b for kind, b in blocks if kind == "item"]
    when = next((b for kind, b in blocks if kind == "when"), "")

    out = ['<section class="part-card">',
           '  <span class="part-num">Part %s</span>' % number,
           "  <h2>%s</h2>" % inline(heading)]
    if when:
        out.append('  <p class="when"><strong>When:</strong> %s</p>' % inline(when))
    if link:
        out.append('  <a class="form-btn" href="%s" target="_blank" '
                   'rel="noopener">Open the Part %s form &#9656;</a>'
                   % (html.escape(link, quote=True), number))
    else:
        out.append('  <span class="form-btn unset">Form link not set yet</span>')
    out.append('  <details class="items">')
    out.append("    <summary>Show all %d questions</summary>" % len(items))

    open_list = False
    for kind, payload in blocks:
        if kind == "item":
            if not open_list:
                out.append('    <ul class="qlist">')
                open_list = True
            out.append(render_item(payload))
            continue
        if open_list:
            out.append("    </ul>")
            open_list = False
        if kind == "group":
            out.append("    <h3>%s</h3>" % inline(payload))
        elif kind == "quote":
            out.append("    <blockquote>%s</blockquote>" % inline(payload))
        elif kind == "para":
            out.append('    <p class="lead">%s</p>' % inline(payload))
    if open_list:
        out.append("    </ul>")
    out.append("  </details>")
    out.append("</section>")
    return out


PAGE = """<!DOCTYPE html>
<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `python study/build_study_page.py`; the source of truth
     is study/questionnaire.md (text) and study/form_links.json (urls). -->
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NPC Interrogation Study &mdash; Questionnaire</title>
  <meta name="description" content="The three-part pilot questionnaire for the NPC interrogation study: background before play, one block after each detective, then comparison and final questions." />
  <link rel="stylesheet" href="../css/style.css" />
  <link rel="stylesheet" href="../css/study.css" />
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="lede">
      Three short forms, one for each moment in the session. Open each one at
      the point described on its card &mdash; later questions are deliberately
      kept out of sight until they apply.
    </p>
    <p class="note">
      Responses are collected in Google Forms. This page is the full
      instrument, readable without a Google account, generated from
      <code>study/questionnaire.md</code> in the
      <a href="{repo}" target="_blank" rel="noopener">project repository</a>.
    </p>
  </header>

  <div class="toolbar">
    <button id="expand-all" type="button">Expand all questions</button>
  </div>

{cards}

  <details class="researcher">
    <summary>For researchers &mdash; protocol, facilitator rules, analysis notes</summary>
{notes}
  </details>

  <footer class="page-foot">
    Generated from <code>study/questionnaire.md</code> by
    <code>study/build_study_page.py</code>. Do not edit this file by hand.
  </footer>

  <script>
    document.getElementById('expand-all').addEventListener('click', function () {{
      var open = this.dataset.state !== 'open';
      Array.prototype.forEach.call(document.querySelectorAll('details'),
        function (d) {{ d.open = open; }});
      this.dataset.state = open ? 'open' : 'closed';
      this.textContent = open ? 'Collapse all' : 'Expand all questions';
    }});
  </script>
</body>
</html>
"""


def main():
    with open(SOURCE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(LINKS, "r", encoding="utf-8") as f:
        links = json.load(f)

    title, front, sections = split_sections(lines)
    title = title.split(" (")[0]  # drop the "(N=10)" suffix in the page header

    cards, notes, total, parts = [], list(front), 0, 0
    for heading, body in sections:
        part = PART_RE.match(heading)
        if not part:
            notes.append("## " + heading)
            notes.extend(body)
            continue
        number, part_title = part.group(1), part.group(2)
        blocks = parse_part(body)
        total += sum(1 for kind, _ in blocks if kind == "item")
        parts += 1
        cards.extend(render_card(number, part_title, blocks,
                                 links.get("part" + number, "")))

    notes_html = ["    " + line for line in render_prose(notes)]
    page = PAGE.format(
        title=html.escape(title),
        repo=REPO_URL,
        cards="\n".join("  " + line for line in cards),
        notes="\n".join(notes_html),
    )

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)

    missing = [k for k in ("part1", "part2", "part3") if not links.get(k)]
    print("Wrote %d questions across %d parts to %s"
          % (total, parts, os.path.relpath(TARGET, REPO_ROOT)))
    if missing:
        print("Form links still empty: %s -- paste the published urls from "
              "createAllParts() into %s and re-run."
              % (", ".join(missing), os.path.relpath(LINKS, REPO_ROOT)))


if __name__ == "__main__":
    main()
