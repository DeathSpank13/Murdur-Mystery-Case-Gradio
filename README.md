# Adaptive NPC Dialogue: State Machines vs. Static Trees

**▶ [Try it in your browser](https://deathspank13.github.io/Murdur-Mystery-Case-Gradio/)** — no install.
The web demo runs all three modes, including a real AI suspect powered by
**Qwen2.5-0.5B-Instruct** loaded entirely client-side via
[transformers.js](https://huggingface.co/docs/transformers.js). It is a
scaled-down stand-in: a 12B model cannot run in a browser, so the demo uses a
small 0.5B model. For the full-quality dynamic mode, run the Python app locally
with Wayfarer-12B (see [Two ways to run](#two-ways-to-run)).

A prototype comparing two ways of running a non player character (NPC) in a
murder mystery interrogation, built as the apparatus for a small user study:

1. **static** a model-free, scripted baseline, in two flavours: a **semantic
   retrieval** lookup the player types at (the input is embedded and matched to
   the nearest pre-written line by meaning), and a standalone **branching
   dialogue** of clickable choices (see below).
2. **dynamic** a Finite State Machine that rewrites a local language model's
   system prompt in real time, so the suspect's persona shifts from Calm to
   Suspicious to Defensive depending on how the player questions her.

The investigator (the player) interrogates Eleanor Vance about the death of a
guest at her dinner party, then decides whether she is guilty. The study
compares player experience and judgement across the two conditions.

The branching dialogue is the second, fuller take on the scripted (static)
approach: a clickable choice tree where some questions can be asked only once and
some choices are mutually exclusive forks. It lives in its own tab and sits
outside the blinded A/B study (clicking versus typing would give the conditions
away); see the design note below for what it demonstrates.

## Project structure

| File | Responsibility |
|------|----------------|
| `fsm.py` | The Finite State Machine: states, transition rules, the per state system prompts, nugget tracking, and the scenario ground truth. |
| `nuggets.py` | The three "slips" (nuggets): trigger topics, per turn drop instructions, reply markers, confrontation definitions, and the confession thresholds. |
| `intent_classifier.py` | Classifies each player turn into a multi-axis `Signal` (evidence, accusation, aggression, warmth, conscience, probing, topic, nugget) with a keyword fallback. |
| `static_dialogue.py` | The control condition. Semantic retrieval (ChromaDB + a Sentence Transformer) over a fixed Q&A database (`data/suspect_qa.json`): the player's input is embedded and the nearest topic's next pre-written variant is returned (repeat questions get "asked and answered" lines). No generation. |
| `branching_dialogue.py` | Standalone choice-based dialogue tree: a menu of clickable options with one-time questions, mutually exclusive forks, and nested follow-ups. |
| `llm_client.py` | Talks to the local llama.cpp server; returns each reply together with its measured latency. |
| `logger.py` | Writes each session (transcript, condition, FSM state, latency, verdicts) to a JSON file under `logs/`. |
| `ui.py` | The Gradio interface: two tabs (the blinded A/B study, and the branching dialogue), researcher view, verdict mechanic, and the per turn logic. |
| `main.py` | Entry point. Checks the server and launches the app. |
| `test_fsm.py` | Lightweight checks for the state machine and verdict scoring. |
| `test_dialogue.py` | Lightweight checks for the branching dialogue engine's once / exclusive / nesting rules. |
| `docs/` | The static, browser-based demo published to GitHub Pages. Plain HTML/CSS/JS ports of the three modes; the AI suspect uses Qwen2.5-0.5B-Instruct via transformers.js. See [The web demo](#the-web-demo). |

## Design notes (worth knowing before a demo or a question)

**Confession is gated by deduction, not pressure.** Eleanor's cover story has
three planted weak points -- the "nuggets" in `nuggets.py`. When the player asks
about the right topics (the wound, when she last saw Charles, how she is holding
up) she lets slip a small detail that contradicts her story: knowing the wound
was to the neck when that was never released, placing herself at the study
doorway inside the murder window, a cut thumb that matches the blood on the
study doorknob. The player must notice a slip in the transcript and confront her
with it; only a landed confrontation makes her realise she is caught (the
one-way awareness boundary), and reaching a confession requires landing **two of
the three**. Invented evidence ("we have your fingerprints") and pure aggression
only make her defensive or hostile -- bullying never produces a confession,
which is the point: it plays like a classic detective story, not an
interrogation by attrition. Each slip is confirmed against her actual reply
(marker substrings) before it counts, so the game state never claims she said
something she didn't.

**Guilt is gated by state.** The base persona holds only her cover story and
says nothing about guilt. The fact that Eleanor is responsible is introduced
only in the aware-band overlays (Resigned onward). A local model told "you are
the killer" on every turn tends to leak it even when relaxed, which would ruin
the interrogation. Gating the secret behind the FSM means the state machine
changes not just her tone but what she conceals; the three slips are the only
controlled cracks, injected one turn at a time.

**The conditions are blinded.** Participants see neutral labels, **Detective A**
and **Detective B**. Which label maps to static or dynamic is randomised per
session and stored only in the log, so testers are not primed toward "the clever
AI one." Turn on **Researcher view** to reveal the live FSM state and the last
response latency during a demo.

**There is a goal and an outcome.** After questioning, the participant submits a
verdict (guilty or innocent) plus a confidence rating. This is logged and scored
against the ground truth, giving a measurable dependent variable: did players
judge correctly more often, and more confidently, in one condition?

**The static baseline is a fixed database, but a fair one.** It is the kind of
scripted suspect games have traditionally used, with the matching done by
*meaning* rather than literal keywords: a fixed Q&A database
(`data/suspect_qa.json`) of topic entries (identity, the party, the guests, the
alibi, money, the weapon, the cellar trip, the blood on the doorknob, and so
on), each holding an ordered list of pre-written reply variants, optional
"asked and answered" repeat lines, and several example questions. The player's
input is embedded with a Sentence Transformer (all-MiniLM-L6-v2) and the
nearest topic's next unspoken variant is returned: the first ask gets the
canonical answer, repeat asks get rephrasings and then pointed
you've-asked-me-this lines, and off-script input gets generic fallbacks that
cycle from polite to testy. The rotation is a deterministic per-session
counter, not a model. This keeps it from being a strawman without turning it
into anything adaptive: it answers obvious questions, tolerates paraphrases the
script never anticipated ("remind me what you're called"), and doesn't parrot
one line forever -- yet it still generates nothing and adapts to nothing. The
three slips are baked into the script as fixed text (the neck detail in the
weapon answer, the doorway at a quarter to ten in the last-seen answer, the cut
thumb in the wellbeing answer -- each spoken once, on the first ask), so the
mystery is technically solvable in both conditions; but the static suspect
never reacts to being confronted, and the guilt fact itself lives only in the
dynamic FSM overlays, never in the database. If a reviewer worries the
comparison is unfair, that is the framing to give: a representative, reasonable
scripted baseline, retrieval-matched and repetition-aware so surface polish
doesn't confound the comparison, not a deliberately broken one. What the
dynamic condition uniquely adds is *reaction*: pressure changes her state, and
a landed confrontation is the only road to a confession.

**The branching dialogue is a separate mode, not part of the study.** The second
tab is a fuller, menu-driven dialogue tree, the kind a real game ships: the
player clicks lines rather than typing. It is kept out of the blinded A/B
comparison on purpose, since clicking versus typing would tell participants
which suspect is which. Its point is to show consequence: some questions can be
asked only **once** then vanish, and some choices are **mutually exclusive
forks** where committing to one hides the paths not taken for the rest of the
run. Topics also nest into follow-up questions. The data lives in
`DIALOGUE_TREE` in `branching_dialogue.py`; `DialogueEngine` enforces the rules
and is unit tested independently of the UI.

## Two ways to run

| | Browser demo | Local Python app |
|---|---|---|
| **Where** | [GitHub Pages link](https://deathspank13.github.io/npc-interrogation/) | Your machine |
| **Install** | Nothing | Python + dependencies (below) |
| **AI model** | Qwen2.5-0.5B-Instruct, in-browser | Wayfarer-12B, local llama.cpp server |
| **Quality** | Lower (small model) | Full |
| **Modes** | All three (static, branching, AI) | All three + the blinded A/B study, logging, and verdict scoring |

The browser demo is the easy way to see the idea. The Python app below is the
real apparatus, with the full-size model and the study instrumentation.

## Setup (local Python app)

1. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   Note: Static mode now uses semantic retrieval (`chromadb` +
   `sentence-transformers`), which pull in `torch` — a large (~GB) install. The
   embedding model (all-MiniLM-L6-v2, ~80 MB) downloads once on first run;
   `main.py` warms it up at startup so the first study turn isn't slow.

3. Install and start the local model server (in a separate terminal):

   ```powershell
   winget install llama.cpp
   llama-server -hf bartowski/Wayfarer-12B-GGUF:Q4_K_M
   ```

   The server listens on port 8080 by default, which is what `llm_client.py`
   expects.

## Run

```powershell
python main.py
```

Open the local URL Gradio prints (usually http://127.0.0.1:7860). The app has
two tabs: **Interrogation (study)** (the blinded A/B comparison) and **Branching
dialogue** (the standalone choice-based mode). Static mode and the branching tab
work even if the llama.cpp model server is not running (Static mode uses its own
local embedding model, not the server); dynamic mode needs the server up.

## The web demo

The `docs/` folder is a self-contained static site (no build step, no server)
published to GitHub Pages. It re-implements the three modes in plain JavaScript:

| File | Mirrors |
|------|---------|
| `docs/js/fsm.js` | `fsm.py` (states, triggers, persona overlays) |
| `docs/js/static_dialogue.js` | `static_dialogue.py` (retrieval; transformers.js embeddings + cosine instead of ChromaDB, over `docs/js/static_qa_data.js`) |
| `docs/js/branching_dialogue.js` | `branching_dialogue.py` (choice tree + engine) |
| `docs/js/llm.js` | `llm_client.py`, but loads Qwen2.5-0.5B-Instruct in-browser via transformers.js |
| `docs/js/app.js`, `docs/index.html`, `docs/css/style.css` | the UI (a JS counterpart to `ui.py`) |

The **Python files are the canonical source**; the JS files are hand-kept mirrors
of the same frozen scenario data and rules (`docs/js/static_qa_data.js` is a
paste of `data/suspect_qa.json`, so the static tab has the same variants,
repeat lines and baked-in slips as the Python app). Note: the web demo's *AI*
tab still mirrors the pre-nugget version of the scenario -- the in-browser 0.5B
model is too small to follow per-turn drop instructions reliably, so the
three-slips confrontation mechanic is a Python-app feature for now. The AI tab
downloads the model on first use (a few hundred MB, then browser-cached) and
runs it on WebGPU where available, falling back to CPU/WASM otherwise.

Preview it locally before publishing:

```powershell
python -m http.server -d docs 8000
```

Then open <http://localhost:8000>. To publish: push the repo to GitHub and set
**Settings → Pages → Source = `main` branch, `/docs` folder**. The site appears
at `https://deathspank13.github.io/npc-interrogation/`.

## Demo script

Tick **Researcher view** so you can watch the state and the nugget tallies
change, select whichever detective is mapped to dynamic (the readout will move
only for that one), and ask these in order:

1. "Good evening. Can you tell me about the party?" stays **Calm**
2. "How exactly was Charles killed?" she slips the **neck** detail
   (researcher view shows `dropped [wound]`)
3. "When did you last see Charles that evening?" she slips the **study
   doorway at a quarter to ten** (`dropped [corridor, wound]`)
4. "We found your fingerprints on the weapon. You killed him." only
   **Defensive** -- invented evidence is mere pressure, she stays unaware
5. "Nobody was told where he was stabbed, and you never saw the body. How do
   you know it was his neck?" confrontation lands, she crosses to
   **Guilty** (aware)
6. "You said you never went down the east corridor -- yet you saw him in the
   study doorway at a quarter to ten." second confrontation, calm deduction
   moves her to **Remorseful**: the confession
7. Anything further settles into **Confessed**

Then question the other detective with the same lines to feel the difference:
the static suspect recites the same fixed lines every session (the three slip
details are baked into its script, spoken once each on the first ask), but it
never registers a confrontation and never confesses -- catching it in a
contradiction changes nothing.

## The session logs (Phase 5 data)

Every session writes `logs/session_<id>.json`, rewritten after each turn so
nothing is lost if a window closes. Each file holds the ordered turns (with
condition, FSM state, and latency in milliseconds) and the verdict(s). This is
the raw material for the comparative analysis.

## Study design notes

- The free A/B switch suits a within subjects demo. For a strict between
  subjects study, assign each participant one detective and hide the switch, or
  counterbalance the order.
- The verdict screen currently reveals the truth. For a within subjects design
  where a participant judges both suspects, consider suppressing that reveal
  until the very end so the first answer does not contaminate the second.
- `SuspectFSM.history` and the per turn latencies give you transition counts and
  response time distributions straight out of the logs.

## Tests

```powershell
python test_fsm.py
python test_dialogue.py
python test_retrieval.py
```

`test_retrieval.py`'s fast checks use a stub embedder (only `chromadb` needed, no
model download) and validate the real database offline (schema rules, and the
three slip markers from `nuggets.py` present exactly once in their carrier
entries); its final integration checks use the real model -- including a sweep
that routes every entry's own example questions back to their entry to catch
cross-topic collisions -- and are skipped automatically if
`sentence-transformers` isn't installed.
