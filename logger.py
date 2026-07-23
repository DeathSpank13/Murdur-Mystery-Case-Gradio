"""
logger.py
=========
Session logging for the user study (Phase 5).

Each interrogation session is written to a timestamped JSON file under logs/.
A session records:
  - which experimental condition was active (static or dynamic) and the neutral
    label the participant saw,
  - every turn: the player's question, the suspect's reply, the FSM state after
    the turn, and the response latency in milliseconds,
  - the participant's verdict(s) and whether each was correct.

This is the raw material for the comparative analysis. Nothing here changes the
interrogation itself; it only observes and persists. The file is rewritten after
every turn, so a session is never lost if a tester closes the window early.
"""

import json
import os
import uuid
from datetime import datetime

LOG_DIR = "logs"


class SessionLogger:
    """Collects one participant session and flushes it to disk on every event."""

    def __init__(self):
        # A sortable, unique id: timestamp plus a short random suffix so two
        # sessions started in the same second cannot collide.
        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        )
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.turns = []
        self.verdicts = []
        os.makedirs(LOG_DIR, exist_ok=True)

    def log_turn(self, condition, label_shown, player_input, reply, state, latency_ms,
                 signal=None, real_latency_ms=None, simulated_latency_ms=None,
                 nuggets_dropped=None, nuggets_confronted=None, nugget_event=None,
                 classifier_latency_ms=None, generation_latency_ms=None,
                 ttft_ms=None, reveal_pre_delay_ms=None, reveal_chars_per_sec=None):
        """
        Record one question and answer exchange.

        ``signal`` is the classified multi-axis reading of the player's turn for
        the dynamic condition (a dict from Signal.as_dict()), or None for the
        static script. It is logged so Phase 5 can analyse not just where the
        suspect ended up but why the FSM moved her there.

        ``nuggets_dropped`` / ``nuggets_confronted`` are the cumulative nugget
        sets after the turn (dynamic only; None for static), and ``nugget_event``
        marks what happened this turn: "drop:<id>", "drop_failed:<id>",
        "confront:<id>", or None. Together they let Phase 5 reconstruct how a
        player found (or missed) the path to a confession.

        ``latency_ms`` is the *perceived* response time the participant felt. For
        the static condition this includes an artificial delay; ``real_latency_ms``
        (the true compute cost) and ``simulated_latency_ms`` (the injected delay)
        are logged separately so the analysis can recover both.

        ``classifier_latency_ms`` / ``generation_latency_ms`` split the dynamic
        condition's real cost into its two model calls (generation includes the
        occasional drop retry). None for static turns and in logs written before
        the split existed.

        ``ttft_ms`` is the generation's time to first streamed token, recorded
        only for live-streamed dynamic turns (None for static turns, buffered
        drop turns, fallback replies, and logs from before streaming existed).
        latency.py harvests it together with the reply length to calibrate the
        static condition's synthesized pacing.

        ``reveal_pre_delay_ms`` / ``reveal_chars_per_sec`` are the *synthetic*
        pacing actually used to reveal the reply: set for static turns (both)
        and for buffered dynamic drop turns (pre-delay 0.0, pace set). None for
        live-streamed turns, whose pacing was the model's own.
        """
        self.turns.append(
            {
                "index": len(self.turns) + 1,
                "condition": condition,            # "static" or "dynamic"
                "label_shown": label_shown,        # e.g. "Detective A"
                "player_input": player_input,
                "npc_reply": reply,
                "fsm_state": state,                # "n/a (static script)" for static
                "signal": signal,                  # classified axes, or None (static)
                "nuggets_dropped": nuggets_dropped,
                "nuggets_confronted": nuggets_confronted,
                "nugget_event": nugget_event,
                "latency_ms": round(latency_ms, 1),  # perceived total
                "real_latency_ms": round(real_latency_ms, 1) if real_latency_ms is not None else None,
                "simulated_latency_ms": round(simulated_latency_ms, 1) if simulated_latency_ms is not None else None,
                "classifier_latency_ms": round(classifier_latency_ms, 1) if classifier_latency_ms is not None else None,
                "generation_latency_ms": round(generation_latency_ms, 1) if generation_latency_ms is not None else None,
                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                "reveal_pre_delay_ms": round(reveal_pre_delay_ms, 1) if reveal_pre_delay_ms is not None else None,
                "reveal_chars_per_sec": round(reveal_chars_per_sec, 1) if reveal_chars_per_sec is not None else None,
            }
        )
        self._flush()

    def log_verdict(self, condition, label_shown, accusation, correct, confidence):
        """Record a participant's accusation about the suspect."""
        self.verdicts.append(
            {
                "condition": condition,
                "label_shown": label_shown,
                "accusation": accusation,          # "guilty" or "innocent"
                "correct": correct,                # bool
                "confidence": confidence,          # 1..5
                "turns_before_verdict": len(self.turns),
            }
        )
        self._flush()

    def _flush(self):
        """Write the full session to its JSON file."""
        path = os.path.join(LOG_DIR, f"session_{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "session_id": self.session_id,
                    "started_at": self.started_at,
                    "turns": self.turns,
                    "verdicts": self.verdicts,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        return path
