// llm.js
// =====
// In-browser language model for the web demo, using transformers.js (ONNX).
//
// This is the browser stand-in for the Python project's local llama.cpp +
// Wayfarer-12B setup (llm_client.py). A 12B model cannot run in a browser, so
// the demo uses Qwen2.5-0.5B-Instruct, which is small enough to download and
// run client-side. The FSM in fsm.js produces exactly the same kind of system
// prompt; we just feed it to this smaller model. Quality is lower than the
// local 12B build by design — that contrast is part of the point.
//
// The model files are fetched from the Hugging Face Hub on first use (a few
// hundred MB) and then cached by the browser. WebGPU is used when available,
// falling back to WASM (CPU) otherwise.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;        // the cached text-generation pipeline
let loadingPromise = null;   // guards against concurrent load() calls

// True if the model has finished loading and is ready to generate.
export function isReady() {
  return generator !== null;
}

// Which backend we will ask transformers.js to use.
export function backend() {
  return navigator.gpu ? "webgpu" : "wasm";
}

// Load the model. `onProgress` receives the raw transformers.js progress events
// ({ status, file, progress, loaded, total, ... }) so the UI can show a bar.
// Safe to call more than once; later calls reuse the in-flight/finished load.
export async function load(onProgress) {
  if (generator) return generator;
  if (loadingPromise) return loadingPromise;

  loadingPromise = pipeline("text-generation", MODEL_ID, {
    dtype: "q4",
    device: backend(),
    progress_callback: onProgress,
  })
    .then((p) => {
      generator = p;
      return p;
    })
    .catch((err) => {
      loadingPromise = null; // allow a retry after a failed load
      throw err;
    });

  return loadingPromise;
}

// Generate one reply. `systemPrompt` comes from SuspectFSM.getSystemPrompt();
// `history` is [{ role: "user"|"assistant", content }, ...] for the prior turns.
// Returns the assistant's text. The pipeline applies the chat template itself.
export async function generate(systemPrompt, history, { maxNewTokens = 200, temperature = 0.7 } = {}) {
  if (!generator) throw new Error("Model is not loaded. Call load() first.");

  const messages = [{ role: "system", content: systemPrompt }, ...history];
  const output = await generator(messages, {
    max_new_tokens: maxNewTokens,
    temperature,
    do_sample: temperature > 0,
  });

  // text-generation with a chat model returns the full message list under
  // generated_text; the last entry is the assistant's new reply.
  return output[0].generated_text.at(-1).content.trim();
}
