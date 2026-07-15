"""
benchmark_llm.py
================
Latency benchmark for the dynamic condition's llama-server setup.

The dynamic condition makes two model calls per player turn (the intent
classifier with its fixed few-shot prefix, then the in-character reply with the
growing conversation), so response time depends heavily on how the server is
launched: context size (`-c`), GPU offload (`-ngl`), flash attention (`-fa`),
KV cache quantisation (`-ctk`/`-ctv`), and parallel slots (`-np`, which lets
the classifier's prefix stay KV-cached in its own slot instead of being
reprocessed every turn). None of those are set by the README's bare launch
command, so this script measures what each actually buys on this machine.

Two modes:

    python benchmark_llm.py --mode attached          # measure the server already on :8080
    python benchmark_llm.py --mode sweep             # launch/kill llama-server per config
    python benchmark_llm.py --mode sweep --full      # extended flag matrix

The workload replays a fixed interrogation (the README demo script plus neutral
fillers). To keep every config comparable, the FSM is driven by the
deterministic ``classify_keywords`` and the transcript uses canned replies, so
the prompts sent to the server are byte-identical across configs; the *real*
classifier call and the *real* generation call still fire every turn and are
what gets timed. The classifier's JSON output is also recorded and diffed
against the first config's, because KV quantisation or flash attention can in
principle shift temperature-0 logits: a diff is a signal to re-run
eval_classifier.py under that config before adopting it.

Run this with the Gradio app CLOSED: in sweep mode the script owns the server
lifecycle and refuses to start if something is already listening on the port.

Results go to benchmarks/results_<timestamp>.json (full detail) and
benchmarks/results.csv (one summary row per config, appended across runs), and
a comparison table is printed at the end.
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests

import intent_classifier
import llm_client
from fsm import SuspectFSM
from nuggets import NUGGETS

DEFAULT_HF_TAG = "bartowski/Wayfarer-12B-GGUF:Q4_K_M"
DEFAULT_PORT = 8080
# First-ever run may download ~7GB via -hf; the model is normally already
# cached, where loading takes well under a minute.
READY_TIMEOUT_S = 600
# One call may be very slow on a badly spilled config; don't let the harness
# die on it, just record the wall time.
CALL_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Server configurations
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """One llama-server launch variant to measure."""
    name: str
    args: list = field(default_factory=list)  # extra llama-server CLI args
    trim_turns: int = 0                       # 0 = full history; N = trim_history(N)


# Small curated matrix (default): each config costs a full model load, so this
# stays at five. baseline_readme is exactly today's README command; its server
# log reveals what the build's auto defaults actually chose.
SMALL_MATRIX = [
    ServerConfig("baseline_readme", []),
    ServerConfig("c4096_all_fa",
                 ["-c", "4096", "-ngl", "all", "-fa", "on"]),
    ServerConfig("c4096_all_fa_kvq8",
                 ["-c", "4096", "-ngl", "all", "-fa", "on",
                  "-ctk", "q8_0", "-ctv", "q8_0"]),
    ServerConfig("c8192_np2_fa_kvq8",
                 ["-c", "8192", "-np", "2", "-ngl", "all", "-fa", "on",
                  "-ctk", "q8_0", "-ctv", "q8_0"]),
    ServerConfig("c8192_np2_fa_kvq8_trim6",
                 ["-c", "8192", "-np", "2", "-ngl", "all", "-fa", "on",
                  "-ctk", "q8_0", "-ctv", "q8_0"],
                 trim_turns=6),
]

# Extended matrix (--full): controls and finer sweeps around the small matrix.
FULL_EXTRA = [
    ServerConfig("c4096_ngl35_fa",
                 ["-c", "4096", "-ngl", "35", "-fa", "on"]),
    ServerConfig("c4096_ngl30_fa",
                 ["-c", "4096", "-ngl", "30", "-fa", "on"]),
    ServerConfig("c4096_ngl25_fa",
                 ["-c", "4096", "-ngl", "25", "-fa", "on"]),
    ServerConfig("c4096_all_fa_off",
                 ["-c", "4096", "-ngl", "all", "-fa", "off"]),
    ServerConfig("c8192_np2_fa_kvf16",
                 ["-c", "8192", "-np", "2", "-ngl", "all", "-fa", "on"]),
    ServerConfig("c8192_np2_kvq8_reuse_trim6",
                 ["-c", "8192", "-np", "2", "-ngl", "all", "-fa", "on",
                  "-ctk", "q8_0", "-ctv", "q8_0", "--cache-reuse", "256"],
                 trim_turns=6),
    ServerConfig("c4096_np1_fa_kvq8",
                 ["-c", "4096", "-np", "1", "-ngl", "all", "-fa", "on",
                  "-ctk", "q8_0", "-ctv", "q8_0"]),
]


# ---------------------------------------------------------------------------
# Deterministic workload
# ---------------------------------------------------------------------------
# The README demo script with two neutral fillers: exercises calm turns, two
# slips (turns 2 and 4 plan drops via classify_keywords topic detection), an
# evidence/accusation turn, and both confrontations (turns 7 and 8).

WORKLOAD_SCRIPT = [
    "Good evening. Can you tell me about the party?",
    "How exactly was Charles killed?",
    "Where were you when the body was found?",
    "When did you last see Charles that evening?",
    "Did anyone else have a reason to hurt him?",
    "We found your fingerprints on the weapon. You killed him.",
    "Nobody was told where he was stabbed, and you never saw the body. "
    "How do you know it was his neck?",
    "You said you never went down the east corridor -- yet you saw him in the "
    "study doorway at a quarter to ten.",
]

# Canned assistant replies keep the transcript identical across configs (the
# real generated reply is timed but discarded). Lengths roughly match real
# replies so history growth is realistic.
CANNED_REPLIES = [
    "It was a lovely evening, at least at first. Charles invited perhaps "
    "thirty of us; there was music in the drawing room and the wine flowed "
    "rather freely. I spent most of my time near the fireplace.",
    "I honestly couldn't tell you the details, inspector. I was told he was "
    "found in the study, and that is really all anyone has said to me. It was "
    "a terrible shock for everyone at the party.",
    "I was in the drawing room with the other guests when the alarm went up. "
    "You can ask any of them; nobody left the room once the commotion "
    "started, and the staff will say the same.",
    "It would have been fairly late in the evening. He seemed in good "
    "spirits, a little distracted perhaps, but nothing out of the ordinary "
    "for Charles when he had business on his mind.",
    "Charles collected grudges the way other men collect stamps, if I am "
    "honest. Half the room owed him money and the other half resented him "
    "for it. I would not know where to begin pointing fingers.",
    "That is absurd. Of course my fingerprints are in that house, I have "
    "visited a hundred times. You are reaching, inspector, and I think you "
    "know it. I had no reason to hurt Charles.",
    "I... someone must have mentioned it. The house was full of whispers that "
    "night, everyone was talking over everyone. I really could not say who "
    "said what to whom, it was all a blur.",
    "I misspoke, that is all. The evening ran together, one moment into the "
    "next. You are twisting a slip of the tongue into something it is not, "
    "and I resent what you are implying.",
]

# When the FSM plans a slip for the reply, the canned transcript must contain
# its marker so commit_reply confirms the drop exactly as in a real session.
CANNED_DROP_SENTENCES = {
    "wound": " Whoever could do that to his neck must have truly hated him.",
    "corridor": (
        " The last I saw of him he was standing in the study doorway at a "
        "quarter to ten, waving me off with the telephone at his ear."
    ),
    "cut": (
        " Forgive the bandage; I caught my thumb on the cellar door latch "
        "last night and it bled terribly for such a small cut."
    ),
}

FILLER_TEMPLATE = (
    "Let's go over the evening once more. Tell me again what you remember, "
    "detail number {i}."
)
FILLER_REPLY = (
    "As I keep telling you, inspector, it was a party like any other: music, "
    "wine, too many people talking at once. I have nothing new to add, "
    "however many times you ask."
)


def build_script(n_turns):
    """The fixed workload, extended with deterministic fillers past turn 8."""
    lines = list(WORKLOAD_SCRIPT)
    replies = list(CANNED_REPLIES)
    for i in range(len(lines), n_turns):
        lines.append(FILLER_TEMPLATE.format(i=i + 1))
        replies.append(FILLER_REPLY)
    return lines[:n_turns], replies[:n_turns]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def base_url(port):
    return f"http://127.0.0.1:{port}"


def port_in_use(port):
    """True if anything answers HTTP on the port (even a still-loading server)."""
    try:
        requests.get(base_url(port) + "/health", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def post_chat(port, payload):
    """
    POST one chat completion. Returns (data, wall_ms, error). ``data`` is the
    decoded JSON on success (including llama-server's non-OpenAI ``timings``
    field when the build provides it), else None with ``error`` set.
    """
    start = time.perf_counter()
    error = None
    data = None
    try:
        resp = requests.post(
            base_url(port) + "/v1/chat/completions",
            json=payload, timeout=CALL_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall_ms = (time.perf_counter() - start) * 1000.0
    return data, wall_ms, error


def classifier_payload(last_npc, player_line):
    """The real classifier request, byte-identical to what classify() sends."""
    messages = intent_classifier._few_shot_messages()
    messages.append({
        "role": "user",
        "content": intent_classifier._user_block(last_npc, player_line),
    })
    return {
        "messages": [{"role": "system",
                      "content": intent_classifier.CLASSIFIER_SYSTEM_PROMPT}]
                    + messages,
        "temperature": 0.0,
        "max_tokens": 80,
        "stream": False,
        "cache_prompt": True,
        "response_format": intent_classifier.RESPONSE_FORMAT,
    }


def reply_payload(system_prompt, context):
    """The in-character reply request. temperature 0 (unlike the app's 0.7) so
    every config generates against identical sampling and stays comparable."""
    return {
        "messages": [{"role": "system", "content": system_prompt}] + context,
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False,
        "cache_prompt": True,
        "repeat_penalty": 1.2,
    }


def call_record(kind, data, wall_ms, error):
    """Flatten one call's result: wall clock plus server timings if present."""
    record = {"kind": kind, "wall_ms": round(wall_ms, 1), "error": error}
    timings = (data or {}).get("timings") or {}
    for key in ("prompt_n", "prompt_ms", "prompt_per_second",
                "predicted_n", "predicted_ms", "predicted_per_second"):
        record[key] = timings.get(key)
    if data:
        try:
            record["content"] = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            record["content"] = None
    else:
        record["content"] = None
    return record


# ---------------------------------------------------------------------------
# Workload runner
# ---------------------------------------------------------------------------

def run_workload(port, n_turns, trim_turns, quiet=False):
    """
    Replay the fixed interrogation against the server on ``port``.

    Returns a list of per-turn dicts, each holding the classifier call record,
    the reply call record, and the FSM bookkeeping needed for the summary.
    """
    script, canned = build_script(n_turns)
    fsm = SuspectFSM()
    history = []
    last_npc = ""
    turns = []

    for index, line in enumerate(script):
        cdata, cwall, cerr = post_chat(port, classifier_payload(last_npc, line))
        classifier_rec = call_record("classifier", cdata, cwall, cerr)

        # Deterministic drive: keyword classification and canned transcript
        # keep prompts identical across configs (see module docstring).
        signal = intent_classifier.classify_keywords(line)
        fsm.transition(signal, line)
        system_prompt = fsm.get_system_prompt()
        pending = fsm.pending_drop

        history.append({"role": "user", "content": line})
        context = llm_client.trim_history(history, trim_turns)
        rdata, rwall, rerr = post_chat(port, reply_payload(system_prompt, context))
        reply_rec = call_record("reply", rdata, rwall, rerr)

        reply_text = canned[index]
        if pending:
            reply_text += CANNED_DROP_SENTENCES[pending]
        dropped = fsm.commit_reply(reply_text)
        history.append({"role": "assistant", "content": reply_text})
        last_npc = reply_text

        turns.append({
            "turn": index + 1,
            "player_line": line,
            "classifier": classifier_rec,
            "reply": reply_rec,
            "fsm_state": fsm.get_state().value,
            "dropped": dropped,
            "confronted": fsm.last_confront,
            "context_messages": len(context) + 1,  # + system
        })
        if not quiet:
            print(
                f"  turn {index + 1}/{len(script)}: "
                f"classifier {cwall:,.0f} ms, reply {rwall:,.0f} ms"
                + (f"  [{cerr or rerr}]" if (cerr or rerr) else "")
            )
    return turns


# ---------------------------------------------------------------------------
# Server lifecycle (sweep mode)
# ---------------------------------------------------------------------------

def launch_server(exe, config, port, hf_tag, log_path):
    """Start llama-server for one config, logging its output to ``log_path``."""
    cmd = [exe, "-hf", hf_tag, "--host", "127.0.0.1", "--port", str(port)]
    cmd += config.args
    handle = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
    return proc, handle


def wait_until_ready(port, proc, timeout_s=READY_TIMEOUT_S):
    """Poll /health until 200. False if the process dies or the timeout hits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(base_url(port) + "/health", timeout=2).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def stop_server(proc, handle):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)
    handle.close()


def parse_server_log(log_path):
    """Pull the offload/context facts and any warnings out of the server log."""
    info = {"offloaded_layers": None, "total_layers": None,
            "n_ctx": None, "n_slots": None, "warnings": []}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return info
    match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    if match:
        info["offloaded_layers"] = int(match.group(1))
        info["total_layers"] = int(match.group(2))
    match = re.search(r"n_ctx\s*=\s*(\d+)", text)
    if match:
        info["n_ctx"] = int(match.group(1))
    match = re.search(r"n_slots\s*=\s*(\d+)", text)
    if match:
        info["n_slots"] = int(match.group(1))
    for line in text.splitlines():
        if re.search(r"warn|not supported|falling back|failed|unable", line,
                     re.IGNORECASE):
            info["warnings"].append(line.strip())
    info["warnings"] = info["warnings"][:12]
    return info


def sample_vram_mb():
    """Total VRAM in use per nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Summaries and output
# ---------------------------------------------------------------------------

def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarize_turns(turns):
    """Condense one workload run into the numbers the comparison table shows."""
    classifier = [t["classifier"] for t in turns]
    replies = [t["reply"] for t in turns]
    warm = classifier[1:] if len(classifier) > 1 else []
    per_turn_walls = [t["classifier"]["wall_ms"] + t["reply"]["wall_ms"]
                      for t in turns]
    return {
        "classifier_cold_ms": classifier[0]["wall_ms"] if classifier else None,
        "classifier_warm_ms": _median([c["wall_ms"] for c in warm]),
        "classifier_warm_prompt_n": _median([c["prompt_n"] for c in warm]),
        "reply_first_ms": replies[0]["wall_ms"] if replies else None,
        "reply_last_ms": replies[-1]["wall_ms"] if replies else None,
        "gen_tps": _median([r["predicted_per_second"] for r in replies]),
        "prompt_tps": _median([r["prompt_per_second"] for r in replies]),
        "turn_median_ms": _median(per_turn_walls),
        "errors": sum(1 for t in turns
                      if t["classifier"]["error"] or t["reply"]["error"]),
    }


def classifier_outputs(turns):
    return [t["classifier"]["content"] for t in turns]


def diff_count(outputs, reference):
    if reference is None:
        return None
    return sum(1 for got, want in zip(outputs, reference) if got != want)


def fmt(value, kind="ms"):
    if value is None:
        return "-"
    if kind == "ms":
        return f"{value:,.0f}"
    if kind == "tps":
        return f"{value:,.1f}"
    return str(value)


TABLE_COLUMNS = [
    ("config", 28, lambda s, x: s["name"]),
    ("turn med ms", 12, lambda s, x: fmt(x["turn_median_ms"])),
    ("cls cold ms", 12, lambda s, x: fmt(x["classifier_cold_ms"])),
    ("cls warm ms", 12, lambda s, x: fmt(x["classifier_warm_ms"])),
    ("warm pN", 8, lambda s, x: fmt(x["classifier_warm_prompt_n"], "n")),
    ("rep t1 ms", 10, lambda s, x: fmt(x["reply_first_ms"])),
    ("rep tN ms", 10, lambda s, x: fmt(x["reply_last_ms"])),
    ("gen t/s", 8, lambda s, x: fmt(x["gen_tps"], "tps")),
    ("pp t/s", 9, lambda s, x: fmt(x["prompt_tps"], "tps")),
    ("VRAM MB", 8, lambda s, x: fmt(s.get("vram_mb"), "n")),
    ("layers", 7, lambda s, x: s.get("offload") or "-"),
    ("cls diff", 8, lambda s, x: fmt(s.get("classifier_diffs"), "n")),
]


def print_table(rows):
    header = "  ".join(name.ljust(width) for name, width, _ in TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        summary = row.get("summary") or {}
        if not summary:
            print(row["name"].ljust(28) + "  LAUNCH FAILED  " + row.get("error", ""))
            continue
        print("  ".join(
            renderer(row, summary).ljust(width)
            for _, width, renderer in TABLE_COLUMNS
        ))


CSV_FIELDS = [
    "run_at", "mode", "name", "args", "trim_turns", "turns",
    "turn_median_ms", "classifier_cold_ms", "classifier_warm_ms",
    "classifier_warm_prompt_n", "reply_first_ms", "reply_last_ms",
    "gen_tps", "prompt_tps", "vram_mb", "offload", "n_ctx", "n_slots",
    "classifier_diffs", "errors", "notes",
]


def append_csv(out_dir, run_at, mode, rows):
    path = os.path.join(out_dir, "results.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            summary = row.get("summary") or {}
            writer.writerow({
                "run_at": run_at,
                "mode": mode,
                "name": row["name"],
                "args": " ".join(row.get("args", [])),
                "trim_turns": row.get("trim_turns", 0),
                "turns": row.get("n_turns"),
                "turn_median_ms": summary.get("turn_median_ms"),
                "classifier_cold_ms": summary.get("classifier_cold_ms"),
                "classifier_warm_ms": summary.get("classifier_warm_ms"),
                "classifier_warm_prompt_n": summary.get("classifier_warm_prompt_n"),
                "reply_first_ms": summary.get("reply_first_ms"),
                "reply_last_ms": summary.get("reply_last_ms"),
                "gen_tps": summary.get("gen_tps"),
                "prompt_tps": summary.get("prompt_tps"),
                "vram_mb": row.get("vram_mb"),
                "offload": row.get("offload"),
                "n_ctx": row.get("n_ctx"),
                "n_slots": row.get("n_slots"),
                "classifier_diffs": row.get("classifier_diffs"),
                "errors": summary.get("errors"),
                "notes": row.get("error", ""),
            })
    return path


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_attached(args, out_dir, run_at):
    if not port_in_use(args.port):
        print(f"No server is answering on port {args.port}. Start llama-server "
              "first, or use --mode sweep to let this script manage it.")
        return []
    print(f"Measuring the server already running on port {args.port} "
          f"(trim={args.trim or 'off'})...")
    turns = run_workload(args.port, args.turns, args.trim)
    row = {
        "name": "attached",
        "args": [],
        "trim_turns": args.trim,
        "n_turns": args.turns,
        "turns": turns,
        "summary": summarize_turns(turns),
        "vram_mb": sample_vram_mb(),
        "classifier_diffs": None,
    }
    return [row]


def run_sweep(args, out_dir, run_at):
    exe = shutil.which("llama-server")
    if not exe:
        print("llama-server was not found on PATH. Install it first "
              "(winget install llama.cpp) or add it to PATH.")
        return []
    if port_in_use(args.port):
        print(f"Something is already listening on port {args.port} -- most "
              "likely the app's own server. Close it first: in sweep mode this "
              "script starts and stops llama-server itself.")
        return []

    configs = list(SMALL_MATRIX) + (list(FULL_EXTRA) if args.full else [])
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as fh:
            configs = [ServerConfig(c["name"], c.get("args", []),
                                    c.get("trim_turns", 0))
                       for c in json.load(fh)]

    rows = []
    reference_outputs = None
    for number, config in enumerate(configs, start=1):
        print(f"\n[{number}/{len(configs)}] {config.name}: "
              f"llama-server {' '.join(config.args) or '(bare)'}"
              + (f"  trim={config.trim_turns}" if config.trim_turns else ""))
        log_path = os.path.join(out_dir, f"server_{run_at}_{config.name}.log")
        row = {"name": config.name, "args": config.args,
               "trim_turns": config.trim_turns, "n_turns": args.turns}
        proc, handle = launch_server(exe, config, args.port, args.hf_tag, log_path)
        try:
            if not wait_until_ready(args.port, proc):
                row["summary"] = None
                row["error"] = "server did not become ready (see log)"
                print("  launch failed; last log lines:")
                stop_server(proc, handle)
                log_info = parse_server_log(log_path)
                for line in log_info["warnings"][-5:]:
                    print(f"    {line}")
                row.update({"offload": None, "n_ctx": None, "n_slots": None,
                            "vram_mb": None, "classifier_diffs": None})
                rows.append(row)
                continue

            turns = run_workload(args.port, args.turns, config.trim_turns)
            row["turns"] = turns
            row["summary"] = summarize_turns(turns)
            row["vram_mb"] = sample_vram_mb()
        finally:
            if proc.poll() is None:
                stop_server(proc, handle)
            elif not handle.closed:
                handle.close()

        log_info = parse_server_log(log_path)
        if log_info["offloaded_layers"] is not None:
            row["offload"] = (f"{log_info['offloaded_layers']}"
                              f"/{log_info['total_layers']}")
        else:
            row["offload"] = None
        row["n_ctx"] = log_info["n_ctx"]
        row["n_slots"] = log_info["n_slots"]
        row["log_warnings"] = log_info["warnings"]

        outputs = classifier_outputs(row["turns"])
        if reference_outputs is None:
            reference_outputs = outputs
            row["classifier_diffs"] = 0
        else:
            row["classifier_diffs"] = diff_count(outputs, reference_outputs)
            if row["classifier_diffs"]:
                print(f"  NOTE: {row['classifier_diffs']} classifier output(s) "
                      "differ from the baseline config. Re-run "
                      "eval_classifier.py under this config before adopting it.")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server configs for the dynamic condition.")
    parser.add_argument("--mode", choices=["attached", "sweep"], default="sweep")
    parser.add_argument("--full", action="store_true",
                        help="sweep the extended config matrix")
    parser.add_argument("--config-file",
                        help="JSON list of {name, args, trim_turns} configs")
    parser.add_argument("--turns", type=int, default=8,
                        help="workload length (default 8; >8 adds fillers)")
    parser.add_argument("--trim", type=int, default=0,
                        help="attached mode: trim history to N turns (0 = off)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hf-tag", default=DEFAULT_HF_TAG)
    parser.add_argument("--out", default="benchmarks",
                        help="results directory (default benchmarks/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "attached":
        rows = run_attached(args, args.out, run_at)
    else:
        rows = run_sweep(args, args.out, run_at)
    if not rows:
        sys.exit(1)

    json_path = os.path.join(args.out, f"results_{run_at}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"run_at": run_at, "mode": args.mode, "hf_tag": args.hf_tag,
                   "turns": args.turns, "results": rows}, fh, indent=2)
    csv_path = append_csv(args.out, run_at, args.mode, rows)

    print("\n=== Comparison ===")
    print_table(rows)
    print(f"\nDetail: {json_path}\nSummary: {csv_path}")
    print("Columns: 'turn med ms' = median (classifier + reply) wall per turn; "
          "'warm pN' = median prompt tokens actually processed on warm "
          "classifier calls (small = the few-shot prefix is being KV-cached); "
          "'cls diff' = turns whose classifier JSON differs from baseline.")


if __name__ == "__main__":
    main()
