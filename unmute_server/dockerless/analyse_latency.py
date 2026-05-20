"""
analyse_latency.py — compare response latency between base and speculative RAG models.

Metrics
-------
* Turn end   : last word end timestamp from input.json (Whisper forced alignment)
* Response onset : first non-silent window in output.wav (energy VAD)
* Latency    : onset − turn_end  (negative = bot started before user finished)

Spec-specific fields parsed from output.timings.json:
  attempt status breakdown, fallback usage, tail-trim, drain timing, continuation.

Usage
-----
    python3.12 dockerless/analyse_latency.py [--items N] [--threshold 0.005]
              [--base-dir PATH] [--spec-dir PATH]
"""

import argparse
import json
import statistics
import wave
from pathlib import Path

import numpy as np


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BASE = Path("results_tmp/fit_rag_benchmark/unmute_rag_gemma3_12b")
DEFAULT_SPEC = Path("results_tmp/fit_rag_benchmark/unmute_anticipate_rag_gemma3_12b")
ONSET_WINDOW_MS   = 30     # sliding window for energy VAD
ONSET_THRESHOLD   = 0.005  # RMS threshold for "audio present"
ONSET_SEARCH_PRE  = 2.0    # seconds before turn_end to start searching output.wav
# ─────────────────────────────────────────────────────────────────────────────


# ── audio helpers ─────────────────────────────────────────────────────────────

def load_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path)) as w:
        sr   = w.getframerate()
        nch  = w.getnchannels()
        sw   = w.getsampwidth()
        raw  = w.readframes(w.getnframes())
    dtype = np.int16 if sw == 2 else np.int32
    scale = 32768.0 if sw == 2 else 2**31
    data  = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale
    if nch > 1:
        data = data.reshape(-1, nch)[:, 0]
    return sr, data


def find_onset(audio: np.ndarray, sr: int,
               search_from: float,
               window_ms: int   = ONSET_WINDOW_MS,
               threshold: float = ONSET_THRESHOLD) -> float | None:
    """Return time (seconds) of first window with RMS > threshold after search_from."""
    hop   = max(1, int(sr * window_ms / 1000))
    start = max(0, int(search_from * sr))
    for i in range(start, len(audio) - hop, hop):
        if float(np.sqrt(np.mean(audio[i:i + hop] ** 2))) > threshold:
            return i / sr
    return None


def turn_end_from_input_json(path: Path) -> float | None:
    """Last word end timestamp from Whisper forced-alignment JSON."""
    try:
        data   = json.loads(path.read_text())
        chunks = data.get("chunks", [])
        if not chunks:
            return None
        ts = chunks[-1].get("timestamp", [None, None])
        return ts[1] if len(ts) > 1 else None
    except Exception:
        return None


# ── spec timing helpers ───────────────────────────────────────────────────────

def _vf(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}s"
    return str(v)


def _ms(v) -> str:
    return f"{v * 1000:+.0f}ms" if v is not None else "N/A"


def spec_summary(sd: dict) -> dict:
    """
    Parse output.timings.json for a speculative run and return a structured
    dict of the most useful diagnostic fields.
    """
    specs  = sd.get("speculation_attempts", [])
    vads   = sd.get("vad_pause_detection_times_sec", [])

    total      = len(specs)
    committed  = [s for s in specs if s.get("status") == "committed"]
    cancelled  = [s for s in specs if s.get("status") == "cancelled"]
    discarded  = [s for s in specs if s.get("status") == "discarded"]
    buffered   = [s for s in specs if s.get("status") == "buffered"]

    # Per-committed attempt detail
    commit_details = []
    for s in committed:
        vad    = s.get("vad_pause_detected_sec") or (
            # fallback: find VAD closest before drain
            next((v for v in reversed(vads)
                  if s.get("committed_prefix_drain_started_sec") is not None
                  and 0 <= (s["committed_prefix_drain_started_sec"] - v) < 2.5), None)
        )
        drain  = s.get("committed_prefix_drain_started_sec")
        drain_vad = (drain - vad) if (drain and vad) else None

        commit_details.append({
            "attempt_index"     : s.get("attempt_index"),
            "trigger_time_sec"  : s.get("trigger_time_sec"),
            "status_reason"     : s.get("status_reason"),
            "generated_text"    : (s.get("generated_text") or "")[:40],
            "buffered_chunks"   : s.get("buffered_audio_chunks", 0),
            "drain_start"       : drain,
            "drain_vad_ms"      : drain_vad * 1000 if drain_vad is not None else None,
            "drain_duration_sec": (
                (s.get("committed_prefix_drain_finished_sec") - s.get("committed_prefix_drain_started_sec"))
                if (s.get("committed_prefix_drain_finished_sec") and s.get("committed_prefix_drain_started_sec"))
                else None
            ),
            "tail_trim"         : s.get("committed_tail_trim_triggered", False),
            "tail_trimmed_sec"  : s.get("committed_tail_trimmed_sec"),
            "continuation_llm_first_token_sec": s.get("continuation_llm_first_token_sec"),
            "continuation_first_audio_sec"    : s.get("continuation_first_non_silent_forwarded_sec"),
            "rag_latency_ms"    : s.get("rag_latency_ms"),
            "rag_score"         : (
                s["rag_top_k_output"][0]["score"]
                if s.get("rag_top_k_output") else None
            ),
            "rag_applied"       : s.get("rag_applied_to_prompt"),
            "rag_reason"        : s.get("rag_not_applied_reason"),
        })

    # Cancelled attempts: distinguish graceful (saved as fallback) vs hard-cancelled (truly wasted)
    wasted = []
    saved_as_fallback = []
    for s in cancelled:
        if s.get("buffered_audio_chunks", 0) <= 0:
            continue
        entry = {
            "attempt_index" : s.get("attempt_index"),
            "trigger_sec"   : s.get("trigger_time_sec"),
            "chunks"        : s.get("buffered_audio_chunks", 0),
            "text"          : (s.get("generated_text") or "")[:30],
            "rag_leaked"    : "[RAG_CONTEXT]" in (s.get("generated_text") or ""),
        }
        if s.get("status_reason") == "graceful_supersede_fallback_saved":
            saved_as_fallback.append(entry)
        else:
            wasted.append(entry)

    # Fallback usage (new field: status_reason == vad_confirmed_from_fallback)
    fallback_used = any(
        s.get("status_reason") == "vad_confirmed_from_fallback" for s in committed
    )

    return {
        "total_attempts"  : total,
        "n_committed"     : len(committed),
        "n_cancelled"     : len(cancelled),
        "n_discarded"     : len(discarded),
        "n_buffered"      : len(buffered),
        "fallback_used"   : fallback_used,
        "commit_details"  : commit_details,
        "wasted_audio"    : wasted,
        "saved_as_fallback": saved_as_fallback,
        "vad_times"       : vads,
    }


# ── per-item analysis ─────────────────────────────────────────────────────────

def analyse_item(item: str, base_dir: Path, spec_dir: Path,
                 threshold: float) -> dict:
    result = {"item": item}

    # --- turn end --------------------------------------------------------
    # prefer spec dir input.json (same file), fallback to base
    for d in (spec_dir / item, base_dir / item):
        ij = d / "input.json"
        if ij.exists():
            result["turn_end"] = turn_end_from_input_json(ij)
            break
    else:
        result["turn_end"] = None

    te = result["turn_end"]

    # --- base model ------------------------------------------------------
    bwav = base_dir / item / "output.wav"
    if bwav.exists() and te is not None:
        sr, audio = load_wav(bwav)
        onset = find_onset(audio, sr, max(0.0, te - ONSET_SEARCH_PRE), threshold=threshold)
        result["base_onset"]   = onset
        result["base_latency"] = (onset - te) if onset is not None else None
    else:
        result["base_onset"]   = None
        result["base_latency"] = None

    # base timing JSON (RAG latency, LLM start for reference)
    btj = base_dir / item / "output.timings.json"
    result["base_timings"] = {}
    if btj.exists():
        bd = json.loads(btj.read_text())
        for g in bd.get("generations", []):
            if g.get("start_trigger") == "vad_pause" and g.get("vad_pause_detected_sec"):
                result["base_timings"] = {
                    "vad"         : g["vad_pause_detected_sec"],
                    "rag_ms"      : g.get("rag_latency_ms"),
                    "llm_start"   : g.get("llm_generation_started_sec"),
                    "tts_first"   : g.get("tts_first_audio_chunk_received_sec"),
                }
                break

    # --- spec model ------------------------------------------------------
    swav = spec_dir / item / "output.wav"
    if swav.exists() and te is not None:
        sr, audio = load_wav(swav)
        onset = find_onset(audio, sr, max(0.0, te - ONSET_SEARCH_PRE), threshold=threshold)
        result["spec_onset"]   = onset
        result["spec_latency"] = (onset - te) if onset is not None else None
    else:
        result["spec_onset"]   = None
        result["spec_latency"] = None

    stj = spec_dir / item / "output.timings.json"
    result["spec_summary"] = {}
    if stj.exists():
        sd = json.loads(stj.read_text())
        result["spec_summary"] = spec_summary(sd)

    return result


# ── formatting ────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    sep = "─" * 110

    # ── 1. Latency table ─────────────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("  RESPONSE LATENCY  (turn_end from input.json Whisper timestamps · onset from energy-VAD on output.wav)")
    print("═" * 110)
    print(f"{'ITEM':<30} {'TURN_END':>9} {'BASE_ONSET':>11} {'BASE_LAT':>10} {'SPEC_ONSET':>11} {'SPEC_LAT':>10}  {'DELTA':>9}")
    print(sep)

    base_lats, spec_lats, deltas = [], [], []
    for r in results:
        te  = r["turn_end"]
        bl  = r["base_latency"]
        sl  = r["spec_latency"]
        dt  = (sl - bl) if (sl is not None and bl is not None) else None

        if bl is not None: base_lats.append(bl)
        if sl is not None: spec_lats.append(sl)
        if dt is not None: deltas.append(dt)

        te_s = f"{te:.2f}s"  if te  is not None else "N/A"
        bo_s = f"{r['base_onset']:.2f}s" if r["base_onset"] is not None else "N/A"
        so_s = f"{r['spec_onset']:.2f}s" if r["spec_onset"] is not None else "N/A"
        bl_s = _ms(bl)
        sl_s = _ms(sl)
        dt_s = _ms(dt)

        print(f"{r['item']:<30} {te_s:>9} {bo_s:>11} {bl_s:>10} {so_s:>11} {sl_s:>10}  {dt_s:>9}")

    print(sep)
    if base_lats and spec_lats:
        def stat_row(label, fn):
            b = fn(base_lats); s = fn(spec_lats)
            d = s - b
            print(f"  {label:<28} {'':>9} {'':>11} {_ms(b):>10} {'':>11} {_ms(s):>10}  {_ms(d):>9}")
        stat_row("MEAN",   statistics.mean)
        stat_row("MEDIAN", statistics.median)
        stat_row("STDEV",  statistics.stdev if len(base_lats) > 1 else lambda x: 0)

    print("\n  Negative latency = bot responded BEFORE the turn ended (early speculative commit)")

    # ── 2. Spec attempt breakdown ─────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("  SPECULATIVE ATTEMPT BREAKDOWN")
    print("═" * 110)
    print(f"{'ITEM':<30} {'TOTAL':>6} {'COMMIT':>7} {'CANCEL':>7} {'DISCARD':>8} {'FALLBK':>7}  COMMITTED DETAILS")
    print(sep)

    for r in results:
        ss = r.get("spec_summary", {})
        if not ss:
            print(f"{r['item']:<30} {'N/A':>6}")
            continue

        fb = "YES" if ss.get("fallback_used") else " no"
        commits = ss.get("commit_details", [])
        commit_str = ""
        if commits:
            parts = []
            for c in commits:
                reason = c["status_reason"] or ""
                short  = "FALLBK" if "fallback" in reason else ("EXPBUF" if "expired" in reason else "VAD")
                txt    = c["generated_text"][:25]
                dv     = f"{c['drain_vad_ms']:+.0f}ms" if c["drain_vad_ms"] is not None else "?"
                trim   = " TRIM" if c["tail_trim"] else ""
                parts.append(f"a{c['attempt_index']}[{short} drain+{dv}{trim} \"{txt}\"]")
            commit_str = "  ".join(parts)
        else:
            commit_str = "(none — normal gen)"

        print(f"{r['item']:<30} {ss['total_attempts']:>6} {ss['n_committed']:>7} {ss['n_cancelled']:>7} "
              f"{ss['n_discarded']:>8} {fb:>7}  {commit_str}")

    # ── 3. Cancelled audio breakdown ──────────────────────────────────────────
    print("\n" + "═" * 110)
    print("  CANCELLED AUDIO BREAKDOWN")
    print("═" * 110)
    any_rows = False
    for r in results:
        ss      = r.get("spec_summary", {})
        saved   = ss.get("saved_as_fallback", [])
        wasted  = ss.get("wasted_audio", [])
        if not saved and not wasted:
            continue
        any_rows = True
        if saved:
            parts = [f"a{w['attempt_index']}(t={w['trigger_sec']}s, {w['chunks']}ch, \"{w['text']}\"{' ⚠RAG_LEAK' if w['rag_leaked'] else ''})"
                     for w in saved]
            print(f"  {r['item']:<30}  SAVED_FALLBACK: {', '.join(parts)}")
        if wasted:
            parts = [f"a{w['attempt_index']}(t={w['trigger_sec']}s, {w['chunks']}ch, \"{w['text']}\"{' ⚠RAG_LEAK' if w['rag_leaked'] else ''})"
                     for w in wasted]
            print(f"  {r['item']:<30}  LOST:           {', '.join(parts)}")
    if not any_rows:
        print("  (none)")

    # ── 4. RAG latency & quality ──────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("  RAG PERFORMANCE  (spec attempts — prefetch stage)")
    print("═" * 110)
    print(f"  {'ITEM':<30} {'COMMIT_ATTEMPT':>14} {'RAG_MS':>8} {'RAG_SCORE':>10} {'APPLIED':>8}  REASON")
    print("  " + sep)
    for r in results:
        for c in r.get("spec_summary", {}).get("commit_details", []):
            rl  = f"{c['rag_latency_ms']:.0f}ms"  if c["rag_latency_ms"]  is not None else "N/A"
            rs  = f"{c['rag_score']:.3f}"          if c["rag_score"]       is not None else "N/A"
            app = "YES" if c["rag_applied"] else "no"
            rsn = c["rag_reason"] or "-"
            print(f"  {r['item']:<30} {'a'+str(c['attempt_index']):>14} {rl:>8} {rs:>10} {app:>8}  {rsn}")

    # ── 5. Continuation timing ────────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("  CONTINUATION TIMING  (for committed spec attempts)")
    print("═" * 110)
    print(f"  {'ITEM':<30} {'ATTEMPT':>8} {'DRAIN_DUR':>10} {'CONT_1stTKN':>13} {'CONT_1stAUD':>13}  TAIL_TRIM")
    print("  " + sep)
    for r in results:
        te = r["turn_end"]
        for c in r.get("spec_summary", {}).get("commit_details", []):
            dd  = f"{c['drain_duration_sec']:.2f}s" if c["drain_duration_sec"] else "N/A"
            ct  = f"{c['continuation_llm_first_token_sec']:.2f}s" if c["continuation_llm_first_token_sec"] else "N/A"
            ca  = f"{c['continuation_first_audio_sec']:.2f}s"     if c["continuation_first_audio_sec"]     else "N/A"
            trim = f"YES ({c['tail_trimmed_sec']:.2f}s trimmed)" if c["tail_trim"] else "no"
            print(f"  {r['item']:<30} {'a'+str(c['attempt_index']):>8} {dd:>10} {ct:>13} {ca:>13}  {trim}")

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir",  default=str(DEFAULT_BASE),
                    help="Results dir for the base (non-speculative) model")
    ap.add_argument("--spec-dir",  default=str(DEFAULT_SPEC),
                    help="Results dir for the speculative RAG model")
    ap.add_argument("--items",     type=int, default=0,
                    help="Analyse only the first N items (0 = all)")
    ap.add_argument("--threshold", type=float, default=ONSET_THRESHOLD,
                    help=f"Energy RMS threshold for response onset detection (default {ONSET_THRESHOLD})")
    ap.add_argument("--category",  default="courses_long_single",
                    help="Item prefix/category to filter")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    spec_dir = Path(args.spec_dir)

    # Collect items present in both dirs
    all_items = sorted(
        p.name for p in spec_dir.iterdir()
        if p.is_dir() and p.name.startswith(args.category)
        and (base_dir / p.name).is_dir()
    )
    if args.items > 0:
        all_items = all_items[: args.items]

    if not all_items:
        print(f"No items found matching '{args.category}' in both dirs.")
        return

    print(f"Analysing {len(all_items)} items  "
          f"[base={base_dir.name}  spec={spec_dir.name}]")

    results = [analyse_item(item, base_dir, spec_dir, args.threshold)
               for item in all_items]

    print_report(results)


if __name__ == "__main__":
    main()
