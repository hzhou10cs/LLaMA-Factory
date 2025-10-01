#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV -> Alpaca JSON/JSONL for LLaMA-Factory (robust session parsing + round-based samples)

INPUT CSV (required columns):
    conversation_id, turn_index, role, text
      - conversation_id: e.g., 'Margaret/6030.F.w2.7.27.17.MD.MP3'
      - role: 'patient' or 'advisor' (final, no 'unknown' please)
      - text: utterance string
      - turn_index: int, order within the conversation

OUTPUTS:
  - alpaca_smartcoaching.jsonl  (one JSON per line)
  - alpaca_smartcoaching.json   (a JSON array)

CLI EXAMPLE (recommended defaults):
  python csv_to_alpaca.py \
    --input_csv source_smartcoaching_final_clean.csv \
    --out_jsonl alpaca_smartcoaching.jsonl \
    --out_json  alpaca_smartcoaching.json \
    --collapse_advisor_runs true \
    --max_history_pairs 8 \
    --prefix_roles \
    --inject_session_in_system false \
    --emit_metadata false
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# =========================
# Config: base system prompt
# =========================
BASE_SYSTEM_PROMPT = (
    "You are a supportive weight-loss coach."
    "Speak naturally and briefly, like a friendly phone call."
    "Offer practical, specific suggestions with numbers and timeframes when helpful."
    "Ask one simple check-in question before you wrap up. Avoid medical advice or diagnosis."
)

# =========================
# Robust session-name parser
# =========================

WEEK_TOKEN_RE   = re.compile(r'^(?:[A-Za-z]*w\.?\d+)$', re.IGNORECASE)  # w12, w.12, Fw10, cAw8
DIGITS_RE       = re.compile(r'^\d+$')
ONE2DIGITS_RE   = re.compile(r'^\d{1,2}$')

def _strip_suffixes(fname: str) -> str:
    """Remove trailing extension; keep base before last dot."""
    return fname.rsplit('.', 1)[0] if '.' in fname else fname

def parse_session_name(name: str) -> Dict[str, Optional[str]]:
    """
    Supports examples like:
      'Margaret/6030.F.w2.7.27.17.MD.MP3'
      'Dominique/6375.w.11.10.5.18.DM.MP3'
      'Dominique/6509.Fw10.4.23.19.DM.MP3'
      'Leland/6170.cA.w12.02.01.18.LB.MP3'

    Extracts: coach, patient_id, week, month, day, hour (strings or None)
    """
    out = {"coach": None, "patient_id": None, "week": None, "month": None, "day": None, "hour": None}
    if not isinstance(name, str) or not name:
        return out

    parts = re.split(r'[\\/]+', name)
    if not parts:
        return out
    if len(parts) == 1:
        coach, rest = None, parts[0]
    else:
        coach, rest = parts[0], parts[-1]
        out["coach"] = coach

    rest = _strip_suffixes(rest)  # strip extension like .MP3
    tokens: List[str] = [t for t in rest.split('.') if t != '']
    if not tokens:
        return out

    # Find first all-digit token as patient_id
    pid_idx = None
    for i, tok in enumerate(tokens):
        if DIGITS_RE.match(tok):
            out["patient_id"] = tok
            pid_idx = i
            break

    # Find week token
    week_idx = None
    start = (pid_idx + 1) if pid_idx is not None else 0
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if WEEK_TOKEN_RE.match(tok):
            m = re.search(r'(\d+)$', tok)
            if m:
                out["week"] = m.group(1)
                week_idx = i
                break
        if tok.lower() == 'w' and i + 1 < len(tokens) and DIGITS_RE.match(tokens[i+1]):
            out["week"] = tokens[i+1]
            week_idx = i
            break
        i += 1

    # Month/Day/Hour as the next 3 one/two-digit tokens after week
    if week_idx is not None:
        found: List[str] = []
        j = week_idx + 1
        while j < len(tokens) and len(found) < 3:
            if ONE2DIGITS_RE.match(tokens[j]):
                found.append(tokens[j])
            j += 1
        if len(found) >= 1: out["month"] = found[0]
        if len(found) >= 2: out["day"]   = found[1]
        if len(found) >= 3: out["hour"]  = found[2]

    return out

# =========================
# Helpers
# =========================

def safe_strip(x: Any) -> str:
    return (str(x).strip()) if isinstance(x, str) else ("" if x is None else str(x))

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

def build_rounds(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build patient_block -> advisor_block rounds after merging same-role adjacency."""
    seq = merge_consecutive_same_roles(turns)
    rounds = []
    i = 0
    while i < len(seq):
        pt_text, adv_text = "", ""
        start_idx = i
        if seq[i]["role"] == "patient":
            pt_text = seq[i]["text"]; i += 1
        if i < len(seq) and seq[i]["role"] == "advisor":
            adv_text = seq[i]["text"]; end_idx = i; i += 1
        else:
            end_idx = i - 1
            rounds.append({"patient_text": pt_text, "advisor_text": adv_text, "start_idx": start_idx, "end_idx": end_idx})
            continue
        rounds.append({"patient_text": pt_text, "advisor_text": adv_text, "start_idx": start_idx, "end_idx": end_idx})
    return rounds

def build_history_pairs(turns: List[Dict[str, Any]], upto_index: int, max_history_pairs: Optional[int], prefix_roles: bool):
    """History = prior patient/assistant pairs before 'upto_index' (exclusive)."""
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
            if user_buf is None: user_buf = ""
            asst_buf = (txt if asst_buf is None else (asst_buf + " " + txt)).strip()
            history.append([user_buf, asst_buf])
            user_buf, asst_buf = None, None
    if user_buf is not None or asst_buf is not None:
        history.append([user_buf or "", asst_buf or ""])
    if isinstance(max_history_pairs, int) and max_history_pairs > 0:
        history = history[-max_history_pairs:]
    return history

def build_system_prompt(base_prompt: str, session: Dict[str, Optional[str]], inject: bool) -> str:
    """
    Returns a system prompt that optionally appends a compact, structured session header.
    Header is intentionally stable (same order & labels) so the model can learn to use it.
    """
    if not inject:
        return base_prompt

    parts = [base_prompt]

    # Build a consistent state header the model can learn to rely on
    # e.g., [SESSION] patient_id=6030 | week=2 | date=7/27 17:00
    header_bits = []
    if session.get("patient_id"):
        header_bits.append(f"patient_id={session['patient_id']}")
    if session.get("week"):
        header_bits.append(f"week={session['week']}")

    # show a coarse timestamp if present (not critical, but helps anchoring)
    date_bits = []
    if session.get("month"): date_bits.append(session["month"])
    if session.get("day"):   date_bits.append(session["day"])
    if session.get("hour"):  date_bits.append(f"{session['hour']}:00")
    if date_bits:
        header_bits.append("date=" + "/".join(date_bits))

    if header_bits:
        parts.append("[SESSION] " + " | ".join(header_bits))

    return " ".join(parts)

# =========================
# Conversion
# =========================

def convert_csv_to_alpaca(
    input_csv: str,
    out_jsonl: str,
    out_json: str,
    session_name_from: str = "conversation_id",
    max_history_pairs: Optional[int] = None,
    prefix_roles: bool = False,
    collapse_advisor_runs: bool = True,   # recommended
    inject_session_in_system: bool = False,
    emit_metadata: bool = False,
):
    df = pd.read_csv(input_csv)
    required = {"conversation_id","turn_index","role","text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns}")

    df = df.sort_values(["conversation_id","turn_index"]).reset_index(drop=True)
    items: List[Dict[str, Any]] = []

    for cid, g in df.groupby("conversation_id", sort=False):
        turns = [{"role": r["role"], "text": safe_strip(r["text"])} for _, r in g.iterrows() if safe_strip(r["text"])]
        if not turns: continue

        session_source = g[session_name_from].iloc[0] if session_name_from in g.columns else cid
        session = parse_session_name(str(session_source))
        sys_prompt = build_system_prompt(BASE_SYSTEM_PROMPT, session, inject=inject_session_in_system)

        if collapse_advisor_runs:
            rounds = build_rounds(turns)
            seq = merge_consecutive_same_roles(turns)
            for r in rounds:
                instruction = r["patient_text"] if r["patient_text"] else "Continue the coaching conversation."
                output = r["advisor_text"]
                if not output: 
                    continue  # skip patient-only blocks
                start_idx_in_seq = r["start_idx"]
                history = build_history_pairs(seq, start_idx_in_seq, max_history_pairs, prefix_roles)
                item = {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "system": sys_prompt,
                    "history": history,
                }
                if emit_metadata:
                    item["metadata"] = {
                        "conversation_id": cid,
                        "session_source": session_source,
                        "session_parsed": session
                    }
                items.append(item)
        else:
            seq = merge_consecutive_same_roles(turns)
            for idx, t in enumerate(seq):
                if t["role"] != "advisor":
                    continue
                instruction = "Continue the coaching conversation."
                for j in range(idx - 1, -1, -1):
                    if seq[j]["role"] == "patient" and seq[j]["text"]:
                        instruction = seq[j]["text"]
                        break
                history = build_history_pairs(seq, idx, max_history_pairs, prefix_roles)
                item = {
                    "instruction": instruction,
                    "input": "",
                    "output": t["text"],
                    "system": sys_prompt,
                    "history": history,
                }
                if emit_metadata:
                    item["metadata"] = {
                        "conversation_id": cid,
                        "session_source": session_source,
                        "session_parsed": session
                    }
                items.append(item)

    # Write files
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {len(items)} Alpaca samples")
    print(f" - JSONL: {out_jsonl}")
    print(f" - JSON:  {out_json}")

# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="CSV → Alpaca JSON/JSONL (round-based; robust session parsing).")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--out_folder", type=str, default="./data/")
    ap.add_argument("--out_jsonl", type=str, default="alpaca_smartcoaching.jsonl")
    ap.add_argument("--out_json", type=str, default="alpaca_smartcoaching.json")
    ap.add_argument("--session_name_from", type=str, default="conversation_id", help="column to parse session info from")
    ap.add_argument("--max_history_pairs", type=int, default=8)
    ap.add_argument("--prefix_roles", action="store_true")
    ap.add_argument("--collapse_advisor_runs", type=str, default="true")
    ap.add_argument("--inject_session_in_system", type=str, default="true", help="if true, append Week/Date to system prompt")
    ap.add_argument("--metadata", type=str, default="false", help="if true, include metadata block per sample")
    return ap.parse_args()

def main():
    args = parse_args()
    collapse = str(args.collapse_advisor_runs).lower() in {"1","true","yes","y"}
    max_hist = args.max_history_pairs if args.max_history_pairs and args.max_history_pairs > 0 else None
    inject = str(args.inject_session_in_system).lower() in {"1","true","yes","y"}
    emit = str(args.metadata).lower() in {"1","true","yes","y"}
    output_json_path = args.out_folder + args.out_json
    output_jsonl_path = args.out_folder + args.out_jsonl
    convert_csv_to_alpaca(
        input_csv=args.input_csv,
        out_jsonl=output_jsonl_path,
        out_json=output_json_path,
        session_name_from=args.session_name_from,
        max_history_pairs=max_hist,
        prefix_roles=bool(args.prefix_roles),
        collapse_advisor_runs=collapse,
        inject_session_in_system=inject,
        emit_metadata=emit,
    )

if __name__ == "__main__":
    main()

