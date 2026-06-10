# Adaptive NPC Dialogue: State Machines vs. Static Trees

**▶ [Try it in your browser](https://deathspank13.github.io/npc-interrogation/)** — no install.
The web demo runs all three modes, including a real AI suspect powered by
**Qwen2.5-0.5B-Instruct** loaded entirely client-side via
[transformers.js](https://huggingface.co/docs/transformers.js). It is a
scaled-down stand-in: a 12B model cannot run in a browser, so the demo uses a
small 0.5B model. For the full-quality dynamic mode, run the Python app locally
with Wayfarer-12B (see [Two ways to run](#two-ways-to-run)).

A prototype comparing two ways of running a non player character (NPC) in a
murder mystery interrogation, built as the apparatus for a small user study:

1. **static** a model-free, scripted baseline, in two flavours: a best-match
   keyword tree the player types at, and a standalone **branching dialogue** of
   clickable choices (see below).
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
| `fsm.py` | The Finite State Machine: states, transition triggers, the per state system prompts, and the scenario ground truth. |
| `static_dialogue.py` | The hardcoded control condition. Best-match keyword scoring over a fixed set of topic branches, scripted replies. No model. |
| `branching_dialogue.py` | Standalone choice-based dialogue tree: a menu of clickable options with one-time questions, mutually exclusive forks, and nested follow-ups. |
| `llm_client.py` | Talks to the local llama.cpp server; returns each reply together with its measured latency. |
| `logger.py` | Writes each session (transcript, condition, FSM state, latency, verdicts) to a JSON file under `logs/`. |
| `ui.py` | The Gradio interface: two tabs (the blinded A/B study, and the branching dialogue), researcher view, verdict mechanic, and the per turn logic. |
| `main.py` | Entry point. Checks the server and launches the app. |
| `test_fsm.py` | Lightweight checks for the state machine and verdict scoring. |
| `test_dialogue.py` | Lightweight checks for the branching dialogue engine's once / exclusive / nesting rules. |
| `docs/` | The static, browser-based demo published to GitHub Pages. Plain HTML/CSS/JS ports of the three modes; the AI suspect uses Qwen2.5-0.5B-Instruct via transformers.js. See [The web demo](#the-web-demo). |

## Design notes (worth knowing before a demo or a question)

**Guilt is gated by state.** The base persona says nothing about guilt. The
fact that Eleanor is responsible is introduced only in the Suspicious and
Defensive overlays. A local model told "you are the killer" on every turn tends
to leak it even when relaxed, which would ruin the interrogation. Gating the
secret behind the FSM means the state machine changes not just her tone but what
she conceals.

**The conditions are blinded.** Participants see neutral labels, **Detective A**
and **Detective B**. Which label maps to static or dynamic is randomised per
session and stored only in the log, so testers are not primed toward "the clever
AI one." Turn on **Researcher view** to reveal the live FSM state and the last
response latency during a demo.

**There is a goal and an outcome.** After questioning, the participant submits a
verdict (guilty or innocent) plus a confidence rating. This is logged and scored
against the ground truth, giving a measurable dependent variable: did players
judge correctly more often, and more confidently, in one condition?

**The static baseline is a fixed tree, but a fair one.** It is the kind of
keyword-matched script games have traditionally used: a finite set of topic
branches (identity, the party, the guests, the alibi, money, the weapon, and so
on), each delivering an ordered list of pre-written lines that the suspect walks
down on repeat visits. Two refinements keep it from being a strawman without
turning it into anything adaptive: the branch set is broad enough to answer
obvious questions like "what is your name?" instead of falling through to a
generic line, and matching is *best-match* rather than first-match, so each
branch is scored by how many and how specific its keywords are and the highest
scorer wins (a long question like "where were you when Charles died?" routes to
the alibi rather than being captured by whichever branch was defined first).
There is still no model and no memory beyond per-topic line counters; the guilt
fact lives only in the dynamic FSM overlays. If a reviewer worries the
comparison is unfair, that is the framing to give: a representative, reasonable
scripted baseline, not a deliberately broken one.

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
work even if the model server is not running; dynamic mode needs it up.

## The web demo

The `docs/` folder is a self-contained static site (no build step, no server)
published to GitHub Pages. It re-implements the three modes in plain JavaScript:

| File | Mirrors |
|------|---------|
| `docs/js/fsm.js` | `fsm.py` (states, triggers, persona overlays) |
| `docs/js/static_dialogue.js` | `static_dialogue.py` (keyword branches) |
| `docs/js/branching_dialogue.js` | `branching_dialogue.py` (choice tree + engine) |
| `docs/js/llm.js` | `llm_client.py`, but loads Qwen2.5-0.5B-Instruct in-browser via transformers.js |
| `docs/js/app.js`, `docs/index.html`, `docs/css/style.css` | the UI (a JS counterpart to `ui.py`) |

The **Python files are the canonical source**; the JS files are hand-kept mirrors
of the same frozen scenario data and rules. The AI tab downloads the model on
first use (a few hundred MB, then browser-cached) and runs it on WebGPU where
available, falling back to CPU/WASM otherwise.

Preview it locally before publishing:

```powershell
python -m http.server -d docs 8000
```

Then open <http://localhost:8000>. To publish: push the repo to GitHub and set
**Settings → Pages → Source = `main` branch, `/docs` folder**. The site appears
at `https://deathspank13.github.io/npc-interrogation/`.

## Demo script

Tick **Researcher view** so you can watch the state change, select whichever
detective is mapped to dynamic (the state readout will move only for that one),
and ask these in order:

1. "Good evening. Can you tell me about the party?" stays **Calm**
2. "Where were you when Charles died?" moves to **Suspicious**
3. "Your story doesn't add up. I think you're lying." stays **Suspicious**
4. "We found your fingerprints on the weapon. You killed him." moves to **Defensive**
5. "I'm sorry, no offense, I'm just asking." steps back to **Suspicious**

Then question the other detective with the same lines to feel the difference.

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
```
