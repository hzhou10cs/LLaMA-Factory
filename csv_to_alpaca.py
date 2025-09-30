#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Smart Coaching CSV (final) to LLaMA-Factory Alpaca format — round-based option.

Input CSV (source_smartcoaching_final.csv):
    conversation_id, turn_index, role, text

Outputs:
  - alpaca_smartcoaching.jsonl  (one JSON per line)
  - alpaca_smartcoaching.json   (a JSON array)

Key options:
- --collapse_advisor_runs true  (recommended): produce ONE sample per round
  where a "round" = consecutive patient block -> consecutive advisor block.
- --collapse_advisor_runs false: produce one sample per advisor utterance (legacy behavior).

Other options kept:
- --max_history_pairs N
- --prefix_roles
- session/patient metadata injection (same as previous upgraded version)
"""

import argparse, json, re
from typing import List, Dict, Any, Optional
import pandas as pd

BASE_SYSTEM_PROMPT = (
    "You are a supportive weight-loss advisor (coach). "
    "Be empathetic, specific, and practical. Use SMART goals, reflect back the patient’s situation, "
    "and suggest concrete next steps. Keep advice safe and non-diagnostic."
)

# ----- Session parsing (same as before) -----
SESSION_RX = re.compile(
    r"(?P<coach>[^/\\]+)[/\\](?P<pid>\d+)\.[A-Za-z]\.w(?P<week>\d+)"
    r"(?:\.(?P<mon>\d{1,2})\.(?P<day>\d{1,2})\.(?P<hour>\d{1,2}))?",
    re.IGNORECASE,
)

def parse_session_name(name: str) -> Dict[str, Optional[str]]:
    if not isinstance(name, str) or not name:
        return {"coach": None, "patient_id": None, "week": None, "month": None, "day": None, "hour": None}
    m = SESSION_RX.search(name)
    if not m:
        return {"coach": None, "patient_id": None, "week": None, "month": None, "day": None, "hour": None}
    gd = m.groupdict()
    return {
        "coach": gd.get("coach"),
        "patient_id": gd.get("pid"),
        "week": gd.get("week"),
        "month": gd.get("mon"),
        "day": gd.get("day"),
        "hour": gd.get("hour"),
    }

def safe_strip(x: Any) -> str:
    return (str(x).strip()) if isinstance(x, str) else ("" if x is None else str(x))

# ----- Role helpers -----
def merge_consecutive_same_roles(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not turns: return turns
    merged = [turns[0].copy()]
    for t in turns[1:]:
        if t["role"] == merged[-1]["role"]:
            merged[-1]["text"] = (merged[-1]["text"] + " " + t["text"]).strip()
        else:
            merged.append(t.copy())
    return merged

def apply_role_prefix(s: str, role: str, prefix_roles: bool) -> str:
    if not prefix_roles:
        return s
    return f"{'Patient' if role=='patient' else 'Advisor'}: {s}"

# ----- Build rounds (patient block -> advisor block) -----
def build_rounds(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
      {
        'patient_text': "...",  # may be empty if advisor started first
        'advisor_text': "...",  # may be empty if patient ends
        'start_idx': int,       # index in 'turns' where patient block starts (or advisor if no patient)
        'end_idx': int          # index where advisor block ends
      }
    """
    # First merge consecutive same-role turns so blocks are clean
    seq = merge_consecutive_same_roles(turns)
    rounds = []
    i = 0
    while i < len(seq):
        # Expect a patient block first in a round; if advisor appears first, we treat patient_text=""
        pt_text = ""
        adv_text = ""
        start_idx = i

        if seq[i]["role"] == "patient":
            pt_text = seq[i]["text"]
            i += 1
        # collect advisor block (if any) immediately following
        if i < len(seq) and seq[i]["role"] == "advisor":
            adv_text = seq[i]["text"]
            end_idx = i
            i += 1
        else:
            # no advisor right after (patient-only block)
            end_idx = i - 1
            rounds.append({"patient_text": pt_text, "advisor_text": adv_text, "start_idx": start_idx, "end_idx": end_idx})
            continue

        # finalize the round with both sides captured
        rounds.append({"patient_text": pt_text, "advisor_text": adv_text, "start_idx": start_idx, "end_idx": end_idx})
    return rounds

# ----- History building (pairs) -----
def build_history_pairs(turns: List[Dict[str, Any]], upto_index: int, max_history_pairs: Optional[int], prefix_roles: bool):
    """
    Build patient/user & advisor/assistant pairs using all turns before 'upto_index' (exclusive),
    merging same-role adjacency for cleaner context.
    """
    prior = merge_consecutive_same_roles(turns[:upto_index])

    history: List[List[str]] = []
    user_buf = None
    asst_buf = None

    for t in prior:
        txt = apply_role_prefix(safe_strip(t["text"]), t["role"], prefix_roles)
        if t["role"] == "patient":
            if user_buf is not None or asst_buf is not None:
                history.append([user_buf or "", asst_buf or ""])
                user_buf, asst_buf = None, None
            user_buf = txt
        else:  # advisor
            if user_buf is None:
                user_buf = ""
            asst_buf = (txt if asst_buf is None else (asst_buf + " " + txt)).strip()
            history.append([user_buf, asst_buf])
            user_buf, asst_buf = None, None

    if user_buf is not None or asst_buf is not None:
        history.append([user_buf or "", asst_buf or ""])

    if isinstance(max_history_pairs, int) and max_history_pairs > 0:
        history = history[-max_history_pairs:]

    return history

# ----- System prompt builder (same idea as before) -----
def build_system_prompt(base_prompt: str, session: Dict[str, Optional[str]], patient_meta: Optional[Dict[str, Any]]) -> str:
    parts = [base_prompt]
    header_bits = []
    if session.get("coach"): header_bits.append(f"Coach: {session['coach']}")
    if session.get("patient_id"): header_bits.append(f"PatientID: {session['patient_id']}")
    if session.get("week"): header_bits.append(f"Week: {session['week']}")
    date_bits = []
    if session.get("month"): date_bits.append(f"{session['month']}")
    if session.get("day"): date_bits.append(f"{session['day']}")
    if session.get("hour"): date_bits.append(f"{session['hour']}:00")
    if date_bits: header_bits.append("Session time-ish: " + "/".join(date_bits))
    if header_bits: parts.append(" | ".join(header_bits))
    if patient_meta:
        meta_bits = []
        for k, v in patient_meta.items():
            vs = safe_strip(v)
            if vs: meta_bits.append(f"{k}={vs}")
        if meta_bits:
            parts.append("Patient profile: " + ", ".join(meta_bits))
    return " ".join(parts)

# ----- Main conversion -----
def convert_csv_to_alpaca(
    input_csv: str,
    out_jsonl: str,
    out_json: str,
    session_name_from: str = "conversation_id",
    meta_csv: Optional[str] = None,
    meta_key: str = "patient_id",
    max_history_pairs: Optional[int] = None,
    prefix_roles: bool = False,
    collapse_advisor_runs: bool = True,   # <—— recommended default
):
    df = pd.read_csv(input_csv)
    required = {"conversation_id","turn_index","role","text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns}")

    # optional metadata
    meta_by_pid = {}
    if meta_csv:
        meta_df = pd.read_csv(meta_csv)
        if meta_key not in meta_df.columns:
            raise ValueError(f"--meta_csv must contain a '{meta_key}' column.")
        meta_by_pid = {
            str(row[meta_key]): {k: row[k] for k in meta_df.columns if k != meta_key}
            for _, row in meta_df.iterrows()
        }

    df = df.sort_values(["conversation_id","turn_index"]).reset_index(drop=True)
    items: List[Dict[str, Any]] = []

    for cid, g in df.groupby("conversation_id", sort=False):
        turns = [{"role": r["role"], "text": safe_strip(r["text"])} for _, r in g.iterrows() if safe_strip(r["text"])]
        if not turns: continue

        session_source = g[session_name_from].iloc[0] if session_name_from in g.columns else cid
        session = parse_session_name(str(session_source))
        patient_id = session.get("patient_id")
        patient_meta = meta_by_pid.get(str(patient_id)) if patient_id else None
        sys_prompt = build_system_prompt(BASE_SYSTEM_PROMPT, session, patient_meta)

        if collapse_advisor_runs:
            # Round-based: each patient_block -> advisor_block becomes one sample
            rounds = build_rounds(turns)
            # We need the history cutoff index for each round. Use the start index of the round.
            seq = merge_consecutive_same_roles(turns)
            for r in rounds:
                # instruction = patient_text if present; else generic
                instruction = r["patient_text"] if r["patient_text"] else "Continue the coaching conversation."
                output = r["advisor_text"]
                # If no advisor text (patient-only block), skip (no target to learn)
                if not output: 
                    continue
                # Build history up to this round start
                start_idx_in_seq = r["start_idx"]
                history = build_history_pairs(seq, start_idx_in_seq, max_history_pairs, prefix_roles)

                items.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "system": sys_prompt,
                    "history": history,
                    "metadata": {
                        "conversation_id": cid,
                        "session_source": session_source,
                        "session_parsed": session
                    }
                })
        else:
            # Legacy: one sample per advisor utterance
            seq = merge_consecutive_same_roles(turns)
            for idx, t in enumerate(seq):
                if t["role"] != "advisor": 
                    continue
                # instruction = last patient utterance before this advisor
                instruction = "Continue the coaching conversation."
                for j in range(idx - 1, -1, -1):
                    if seq[j]["role"] == "patient" and seq[j]["text"]:
                        instruction = seq[j]["text"]
                        break
                history = build_history_pairs(seq, idx, max_history_pairs, prefix_roles)
                items.append({
                    "instruction": instruction,
                    "input": "",
                    "output": t["text"],
                    "system": sys_prompt,
                    "history": history,
                    "metadata": {
                        "conversation_id": cid,
                        "session_source": session_source,
                        "session_parsed": session
                    }
                })

    # Write
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {len(items)} Alpaca samples")
    print(f" - JSONL: {out_jsonl}")
    print(f" - JSON:  {out_json}")

def parse_args():
    ap = argparse.ArgumentParser(description="CSV → Alpaca JSON/JSONL (rounds or per-utterance).")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, default="alpaca_smartcoaching.jsonl")
    ap.add_argument("--out_json", type=str, default="alpaca_smartcoaching.json")
    ap.add_argument("--session_name_from", type=str, default="conversation_id")
    ap.add_argument("--meta_csv", type=str, default="")
    ap.add_argument("--meta_key", type=str, default="patient_id")
    ap.add_argument("--max_history_pairs", type=int, default=0)
    ap.add_argument("--prefix_roles", action="store_true")
    ap.add_argument("--collapse_advisor_runs", type=str, default="true")
    return ap.parse_args()

def main():
    args = parse_args()
    max_hist = args.max_history_pairs if args.max_history_pairs and args.max_history_pairs > 0 else None
    collapse = str(args.collapse_advisor_runs).lower() in {"1","true","yes","y"}
    convert_csv_to_alpaca(
        input_csv=args.input_csv,
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        session_name_from=args.session_name_from,
        meta_csv=(args.meta_csv or None),
        meta_key=args.meta_key,
        max_history_pairs=max_hist,
        prefix_roles=bool(args.prefix_roles),
        collapse_advisor_runs=collapse,
    )

if __name__ == "__main__":
    main()
