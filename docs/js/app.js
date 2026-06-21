// app.js
// =====
// UI wiring for the three-tab browser demo. The dialogue logic lives in the
// ported modules (fsm.js, static_dialogue.js, branching_dialogue.js) and the
// model in llm.js; this file only turns their state into DOM updates.

import { SuspectFSM } from "./fsm.js";
import * as staticDialogue from "./static_dialogue.js";
import { DialogueEngine, BACK_ID } from "./branching_dialogue.js";
import * as llm from "./llm.js";
import { classifyIntent } from "./intent.js";

// --- tiny DOM helpers -------------------------------------------------------
const $ = (id) => document.getElementById(id);

function addMessage(container, role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// --- tab switching ----------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

// ===========================================================================
// AI suspect tab
// ===========================================================================
let fsm = new SuspectFSM();
let aiHistory = []; // [{ role, content }, ...] for the model

const backendName = $("backend-name");
backendName.textContent = llm.backend() === "webgpu" ? "WebGPU (fast)" : "WASM / CPU (slower)";

const aiInput = $("ai-input");
const aiSend = $("ai-send");

function setAiEnabled(on) {
  aiInput.disabled = !on;
  aiSend.disabled = !on;
}

$("load-model-btn").addEventListener("click", async () => {
  const btn = $("load-model-btn");
  const wrap = $("progress-wrap");
  const bar = $("progress-bar");
  const text = $("progress-text");
  btn.disabled = true;
  btn.textContent = "Loading…";
  wrap.hidden = false;

  try {
    await llm.load((p) => {
      // transformers.js emits per-file download events plus status updates.
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        bar.style.width = `${pct}%`;
        const mb = (p.loaded / 1e6).toFixed(0);
        const totMb = (p.total / 1e6).toFixed(0);
        text.textContent = `${p.file || "model"} — ${mb}/${totMb} MB`;
      } else if (p.status === "ready" || p.status === "done") {
        text.textContent = "Ready";
      }
    });
    bar.style.width = "100%";
    text.textContent = "Model ready";
    btn.textContent = "Model loaded";
    setAiEnabled(true);
    if ($("ai-chat").children.length === 0) {
      addMessage($("ai-chat"), "npc",
        "Good evening, Inspector. Ask me whatever you need — it has been a dreadful day.");
    }
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.textContent = "Load AI model";
    text.textContent = "Load failed — see console. Try a Chromium browser for WebGPU.";
  }
});

// Researcher view toggle
const researcher = $("researcher-view");
const stateReadout = $("state-readout");
const latencyReadout = $("latency-readout");
researcher.addEventListener("change", () => {
  const on = researcher.checked;
  stateReadout.hidden = !on;
  latencyReadout.hidden = !on;
});

$("ai-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = aiInput.value.trim();
  if (!question || !llm.isReady()) return;

  addMessage($("ai-chat"), "player", question);
  aiInput.value = "";
  setAiEnabled(false);

  const thinking = addMessage($("ai-chat"), "npc thinking", "…");
  const start = performance.now();
  try {
    // Classify the turn on its multi-axis Signal first (using the history so far
    // for context), then move the FSM on the combination of axes. This is a
    // separate model call, so it adds to the turn latency on the small model.
    const signal = await classifyIntent(question, aiHistory);
    fsm.transition(signal);
    const systemPrompt = fsm.getSystemPrompt();
    aiHistory.push({ role: "user", content: question });

    const reply = await llm.generate(systemPrompt, aiHistory);
    const latency = performance.now() - start;
    thinking.classList.remove("thinking");
    thinking.textContent = reply;
    aiHistory.push({ role: "assistant", content: reply });

    // Researcher readout: state plus the axes that drove it, so a demo can show
    // why she moved, not just where she landed.
    const axes = Object.entries(signal)
      .map(([key, value]) => `${key}=${value}`)
      .join(", ");
    const awareTag = fsm.isAware() ? " (aware)" : "";
    stateReadout.textContent = `State: ${fsm.getState()}${awareTag} — ${axes}`;
    latencyReadout.textContent = `Latency: ${latency.toFixed(0)} ms`;
  } catch (err) {
    console.error(err);
    thinking.classList.remove("thinking");
    thinking.textContent = "[The suspect says nothing. The model errored — see console.]";
  } finally {
    setAiEnabled(true);
    aiInput.focus();
  }
});

$("ai-reset").addEventListener("click", () => {
  fsm = new SuspectFSM();
  aiHistory = [];
  $("ai-chat").innerHTML = "";
  stateReadout.textContent = "";
  latencyReadout.textContent = "";
  if (llm.isReady()) {
    addMessage($("ai-chat"), "npc",
      "Good evening, Inspector. Ask me whatever you need — it has been a dreadful day.");
  }
});

// ===========================================================================
// Static retrieval tab
// ===========================================================================
// Retrieval embeds the question with a small model, so the first reply waits on
// a ~25 MB model download (then browser-cached). getResponse() is async; we show
// a "thinking…" bubble while it resolves, like the AI tab does.
let staticCounts = {};

$("static-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("static-input");
  const question = input.value.trim();
  if (!question) return;
  addMessage($("static-chat"), "player", question);
  input.value = "";

  const thinking = addMessage($("static-chat"), "npc thinking", "…");
  try {
    const reply = await staticDialogue.getResponse(question, staticCounts);
    thinking.classList.remove("thinking");
    thinking.textContent = reply;
  } catch (err) {
    console.error(err);
    thinking.classList.remove("thinking");
    thinking.textContent = "[The suspect says nothing. The retrieval model failed to load — see console.]";
  } finally {
    input.focus();
  }
});

$("static-reset").addEventListener("click", () => {
  staticCounts = {};
  $("static-chat").innerHTML = "";
});

// ===========================================================================
// Branching dialogue tab
// ===========================================================================
let engine = new DialogueEngine();

function renderBranching() {
  const chat = $("branch-chat");
  chat.innerHTML = "";
  for (const { speaker, text } of engine.transcript) {
    addMessage(chat, speaker === "player" ? "player" : "npc", text);
  }

  const optionsEl = $("branch-options");
  optionsEl.innerHTML = "";
  for (const opt of engine.availableOptions()) {
    const btn = document.createElement("button");
    btn.textContent = opt.text;
    if (opt.id === BACK_ID) btn.classList.add("back");
    // Dim repeatable options the player has already chosen, game-style.
    if (engine.chosen.has(opt.id)) btn.classList.add("option-used");
    btn.addEventListener("click", () => {
      engine.choose(opt.id);
      renderBranching();
    });
    optionsEl.appendChild(btn);
  }
}

$("branch-reset").addEventListener("click", () => {
  engine = new DialogueEngine();
  renderBranching();
});

// Initial render of the branching menu on load.
renderBranching();
