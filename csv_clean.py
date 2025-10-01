#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged pipeline for Smart Coaching transcripts:

1) Read original CSV (robust encoding).
2) Extract segments and assign roles (heuristics; optional LLM fallback).
3) Immediately handle 'unknown':
   - Drop trivial one/two-word 'unknown' acks.
   - Classify the remaining 'unknown' via a small LLM.
     * Uses a streaming Dataset → pipeline iterator (one call) to maximize GPU efficiency.
     * Shows a tqdm bar ONLY for this slow step.
4) Remove timestamps and end-of-call summaries for the clean export.
5) Write:
   - smartcoaching_roles.csv  (with original speaker + final role, audit trail)
   - smartcoaching_clean.csv  (final role, cleaned text; no 'unknown')

Usage example:
  python read_csv_merged.py \
    --input_csv SmartCoachingCalls.csv \
    --out_roles smartcoaching_roles.csv \
    --out_clean smartcoaching_clean.csv \
    --use_llm true \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --llm_batch_size 16
"""
import argparse
import ast
import csv
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

# Optional streaming imports (for maximum GPU efficiency).
# If unavailable, we fall back to a simple chunked loop.
_DATASETS_OK = True
try:
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
except Exception:
    _DATASETS_OK = False

# ---------------- Heuristic cues ----------------

ADVISOR_PATTERNS = [
    r"\bwe are (officially )?in week\s*\d+\b",
    r"\bwere you able to (get to|complete|finish) lesson\b",
    r"\b(looking|look) at your data\b",
    r"\byour (previous )?smart goal\b",
    r"\b(today|this week) (we|we'll|we will)\b",
    r"\b(let's|let us) (set|pick|talk about) (a )?goal\b",
    r"\byou did (a total of )?\d+ (minutes|min)\b",
    r"\b(remind you|as a reminder)\b",
    r"\byou hit your (300|[12]00) (minute|min) goal\b",
    r"\byour (goal|target) (is|was)\b",
    r"\byour weight (is|was)\b",
    r"\byour (food|diet|intake|tracking|log)\b",
    r"\blesson\s*\d+\b",
    r"\bmodule\s*\d+\b",
    r"\baudio recorded|consent to be recorded\b",
    r"\bcoach\b",
]

PATIENT_PATTERNS = [
    r"\bi (did|have|was|will|am|feel|think|couldn'?t|didn'?t)\b",
    r"\bmy (weight|goal|diet|intake|schedule|doctor|knees|back)\b",
    r"\bi (ate|walked|ran|exercised|tracked)\b",
    r"\bi (forgot|missed|skipped)\b",
    r"\bi (could|can) (try|do)\b",
]

def count_matches(patterns: List[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))

# ---------------- Trivial unknown acks to drop ----------------

TRIVIAL_ACKS = {
    "ok","okay","okey","okey-dokey",
    "yes","yeah","yep","yup","uh-huh","mm-hmm","mmhmm","mm hmm","mhm",
    "thanks","thank you","thx","ty","tysm",
    "sure","right","alright","all right",
    "hmm","uh","um","mmm","huh",
    "great","cool","nice","awesome","perfect","fine","good","got it","understood",
    "i see","sounds good","makes sense","indeed","exactly","correct"
}

def _norm_for_match(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_trivial_unknown(text: str, drop_max_chars: int) -> bool:
    t = (text or "").strip()
    if len(t) <= drop_max_chars:
        nm = _norm_for_match(t)
        if nm in TRIVIAL_ACKS or nm.replace("'", "") in TRIVIAL_ACKS:
            return True
        if len(nm.split()) <= 2 and nm in {"ok","okay","yes","yeah","yep","thanks","thank you","sure","right","alright"}:
            return True
    return False

# ---------------- Load & extract segments ----------------

def load_results_cell(cell: Any) -> Any:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, (dict, list)):
        return cell
    text = str(cell).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None

def extract_segments(results_obj: Any) -> List[Dict[str, Any]]:
    segs: List[Dict[str, Any]] = []
    if isinstance(results_obj, dict):
        base = results_obj.get("segments")
        if not isinstance(base, list):
            base = None
            for v in results_obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "text" in v[0]:
                    base = v
                    break
        if base is None:
            return segs
    elif isinstance(results_obj, list):
        base = results_obj
    else:
        return segs

    for i, item in enumerate(base):
        if not isinstance(item, dict):
            continue
        speaker = (
            item.get("speaker_label")
            or item.get("speaker")
            or item.get("SPEAKER")
            or item.get("label")
            or item.get("speaker_id")
            or ""
        )
        text = item.get("text") or item.get("TEXT") or item.get("utterance") or ""
        start = item.get("start")
        end = item.get("end")
        if not str(text).strip():
            continue
        segs.append({"idx": i, "speaker": str(speaker), "text": str(text), "start": start, "end": end})
    return segs

# ---------------- Heuristic mapping ----------------

def heuristic_role_scores(segments: List[Dict[str, Any]], k_first:int=12) -> Dict[str, Dict[str, int]]:
    scores: Dict[str, Dict[str, int]] = {}
    for seg in segments[:max(k_first, 1)]:
        spk = seg["speaker"]
        txt = seg["text"]
        d = scores.setdefault(spk, {"advisor": 0, "patient": 0})
        d["advisor"] += count_matches(ADVISOR_PATTERNS, txt)
        d["patient"] += count_matches(PATIENT_PATTERNS, txt)
    return scores

def decide_mapping_by_heuristics(segments: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], bool]:
    scores = heuristic_role_scores(segments, k_first=14)
    counts: Dict[str, int] = {}
    for s in segments[:40]:
        sp = (s["speaker"] or "").strip()
        if not sp or sp.upper() == "UNKNOWN":
            continue
        counts[sp] = counts.get(sp, 0) + 1
    candidates = [sp for sp, _ in sorted(counts.items(), key=lambda x: -x[1])][:2]
    if len(candidates) < 2:
        return None, None, False

    def adv_score(spk):
        sc = scores.get(spk, {"advisor":0,"patient":0})
        return sc["advisor"] - sc["patient"]

    ranked = sorted(candidates, key=lambda s: adv_score(s), reverse=True)
    adv_spk, pat_spk = ranked[0], ranked[1]
    raw_adv = scores.get(adv_spk, {}).get("advisor", 0)
    gap = adv_score(adv_spk) - adv_score(pat_spk)
    confident = (gap >= 2) or (raw_adv >= 2)
    return adv_spk, pat_spk, confident

# ---------------- Small LLM ----------------

class SmallLLM:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: Optional[str] = None, max_new_tokens: int = 64, batch_size:int=16):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._pipe = None

    def _lazy_init(self):
        if self._pipe is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if tok.pad_token_id is None:
                if tok.eos_token_id is not None:
                    tok.pad_token = tok.eos_token
                    tok.pad_token_id = tok.eos_token_id
                else:
                    tok.add_special_tokens({"pad_token": "<|pad|>"})

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto" if self.device in (None, "auto") else None,
            )

            if model.config.pad_token_id is None and tok.pad_token_id is not None:
                model.config.pad_token_id = tok.pad_token_id
            if getattr(model, "resize_token_embeddings", None) and len(tok) != model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(tok))

            tok.padding_side = "left"
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                batch_size=self.batch_size,
            )

    def _normalize_generated(self, outs: Any) -> List[str]:
        result: List[str] = []
        if isinstance(outs, list):
            if not outs:
                return result
            if isinstance(outs[0], dict):
                for o in outs:
                    result.append(o.get("generated_text", o.get("summary_text", "")))
            elif isinstance(outs[0], list):
                for sub in outs:
                    if sub and isinstance(sub[0], dict):
                        result.append(sub[0].get("generated_text", sub[0].get("summary_text", "")))
                    else:
                        result.append("")
            else:
                for _ in outs:
                    result.append("")
        else:
            result.append("")
        return result

    def infer_many(self, prompts: List[str]) -> List[str]:
        self._lazy_init()
        outs = self._pipe(prompts, return_full_text=False)
        return self._normalize_generated(outs)

    def map_speakers_many(self, previews: List[str]) -> List[Optional[Tuple[str,str]]]:
        self._lazy_init()
        prompts = []
        for conv_preview in previews:
            p = (
                "You are labeling a weight-loss coaching call with two roles: advisor (coach) and patient.\n"
                "Given the following first few utterances, decide whether SPEAKER_00 or SPEAKER_01 is the advisor.\n"
                "Output strictly one line in JSON like {\"advisor\":\"SPEAKER_01\",\"patient\":\"SPEAKER_00\"}.\n\n"
                f"{conv_preview}\n\nAnswer:"
            )
            prompts.append(p)
        raw = self.infer_many(prompts)
        results: List[Optional[Tuple[str,str]]] = []
        for r in raw:
            m = re.search(r"\{.*\}", r, flags=re.DOTALL)
            if not m:
                results.append(None)
                continue
            try:
                obj = json.loads(m.group(0))
                adv = obj.get("advisor")
                pat = obj.get("patient")
                if adv and pat and adv != pat:
                    results.append((adv, pat))
                else:
                    results.append(None)
            except Exception:
                results.append(None)
        return results

# ---------------- Cleaning helpers ----------------

def looks_like_summary(text: str) -> bool:
    txt = (text or "").strip().lower()
    if not txt:
        return False
    if any(kw in txt for kw in [
        "summary:", "in summary", "to summarize", "this week you will", "overall it seems",
        "overall,", "recap:", "as a recap", "coach summary", "call summary"
    ]):
        return True
    if len(txt.split()) > 150 and re.search(r"\byou (will|should|are going to)\b", txt):
        return True
    return False

def normalize_text(text: str) -> str:
    t = (text or "")
    t = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", t)
    t = re.sub(r"\(\d{1,2}:\d{2}(?::\d{2})?\s*-\s*\d{1,2}:\d{2}(?::\d{2})?\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------- Main pipeline ----------------

def process_file(
    input_csv: str,
    out_roles: str,
    out_clean: str,
    use_llm: bool = False,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    llm_batch_size: int = 16,
    window_left: int = 4,
    window_right: int = 2,
    drop_max_chars: int = 12,
    verbose: bool = True,
):
    # 1) Read original CSV with robust encoding (print-only)
    reformatted_file = '__reformatted_utf8.csv'
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            with open(input_csv, 'r', newline='', encoding=enc, errors='strict') as f_in, \
                 open(reformatted_file, 'w', newline='', encoding='utf-8') as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                for row in reader:
                    writer.writerow(row)
            print(f"[read] Decoded with encoding={enc} → re-wrote as UTF-8: {reformatted_file}")
            break
        except UnicodeDecodeError as e:
            print(f"[read] Decode failed with {enc}: {e}")
    else:
        raise RuntimeError("Could not decode the file with common encodings.")

    # 2) Identify columns (print-only)
    import pandas as pd
    df = pd.read_csv(reformatted_file)
    path_col = next((c for c in df.columns if str(c).strip().lower() in {"relative_path", "path", "file", "filename"}), df.columns[0])
    results_col = next((c for c in df.columns if str(c).strip().lower() in {"results", "json", "segments"}), None)
    if results_col is None:
        raise ValueError("Could not locate a RESULTS/segments/JSON column in the CSV. Found: " + ", ".join(df.columns))
    print(f"[cols] path_col='{path_col}', results_col='{results_col}'")

    llm = SmallLLM(model_id=model_id, device=device, max_new_tokens=max_new_tokens, batch_size=llm_batch_size) if use_llm else None

    # 3) Parse segments + heuristic mapping (print-only)
    all_convs: Dict[str, List[Dict[str,Any]]] = {}
    mapping: Dict[str, Dict[str, Any]] = {}
    need_llm_mapping: List[str] = []

    for ridx, row in df.iterrows():
        conv_id = str(row.get(path_col, f"row_{ridx}"))
        results_obj = load_results_cell(row.get(results_col))
        segments = extract_segments(results_obj)
        if not segments:
            if verbose:
                print(f"[warn] No segments for {conv_id}")
            continue
        all_convs[conv_id] = segments
        adv_spk, pat_spk, confident = decide_mapping_by_heuristics(segments)
        mapping[conv_id] = {"adv_spk": adv_spk, "pat_spk": pat_spk, "confident": bool(confident), "method": "heuristic"}
        if use_llm and not confident:
            need_llm_mapping.append(conv_id)

    # Optional: LLM mapping for speaker identities (no tqdm here)
    if use_llm and need_llm_mapping:
        print("[mapping] Use LLM to correct the speaker identities ...")
        previews, ids = [], []
        for cid in need_llm_mapping:
            segs = all_convs[cid][:14]
            preview = "\n".join([f'{s["speaker"]}: {s["text"]}' for s in segs])
            previews.append(preview)
            ids.append(cid)
        if previews:
            raw = llm.map_speakers_many(previews)
            for cid, res in zip(ids, raw):
                if res is not None:
                    adv_spk, pat_spk = res
                    mapping[cid] = {"adv_spk": adv_spk, "pat_spk": pat_spk, "confident": True, "method": "llm"}

    # 4) Build role_rows and clean_rows (print-only)
    print("[create] Genearating the output .csv file (role.csv and clean.csv) ...")
    role_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []

    for conv_id, segments in all_convs.items():
        info = mapping.get(conv_id, {"adv_spk": None, "pat_spk": None, "confident": False, "method": "fallback"})
        adv_spk, pat_spk, confident, method = info["adv_spk"], info["pat_spk"], info["confident"], info["method"]
        if adv_spk is None or pat_spk is None:
            counts = {}
            for s in segments:
                sp = (s["speaker"] or "").strip()
                if not sp or sp.upper() == "UNKNOWN":
                    continue
                counts[sp] = counts.get(sp, 0) + 1
            common = sorted(counts.items(), key=lambda x: -x[1])
            if len(common) >= 2:
                adv_spk, pat_spk = common[0][0], common[1][0]
                method = "fallback"

        for t, seg in enumerate(segments):
            spk = (seg["speaker"] or "").strip()
            text = seg["text"]
            start = seg.get("start")
            end = seg.get("end")

            if not spk or spk.upper() == "UNKNOWN":
                role = "unknown"
            else:
                role = "advisor" if spk == adv_spk else ("patient" if spk == pat_spk else "unknown")

            role_rows.append({
                "conversation_id": conv_id,
                "turn_index": t,
                "speaker_original": spk,
                "role": role,
                "text": text,
                "start": start,
                "end": end,
                "mapping_confident": bool(confident),
                "mapping_method": method,
            })

            if not looks_like_summary(text):
                clean_rows.append({
                    "conversation_id": conv_id,
                    "turn_index": t,
                    "role": role,
                    "text": normalize_text(text),
                })

    # 5A) Drop trivial-ack unknowns (print-only)
    before_unknown = sum(1 for r in clean_rows if r["role"] == "unknown")
    drop_idx = [i for i, r in enumerate(clean_rows) if r["role"] == "unknown" and is_trivial_unknown(r["text"], drop_max_chars=drop_max_chars)]
    if drop_idx:
        print(f"[Unknown clean] Dropping {len(drop_idx)} trivial-ack 'unknown' lines from clean output (before unknowns: {before_unknown}).")
        for i in reversed(drop_idx):
            clean_rows.pop(i)

    # 5B) Classify remaining unknowns via LLM (STREAMED; tqdm shown here only)
    if use_llm:
        print("[Unknown clean] LLM Inferencing the remaining unknowns via context ...")
        df_clean = pd.DataFrame(clean_rows).sort_values(["conversation_id", "turn_index"]).reset_index(drop=True)
        contexts: List[str] = []
        positions: List[Tuple[str,int]] = []

        for cid, g in df_clean.groupby("conversation_id", sort=False):
            g = g.sort_values("turn_index").reset_index(drop=True)
            for _, row in g.iterrows():
                if str(row["role"]).lower() != "unknown":
                    continue
                t = int(row["turn_index"])
                left = g[(g["turn_index"] >= t - window_left) & (g["turn_index"] < t)]
                right = g[(g["turn_index"] > t) & (g["turn_index"] <= t + window_right)]
                lines = []
                for _, r in left.iterrows():
                    lines.append(f'{r["role"]}: {str(r["text"]).strip()}')
                lines.append(f'[UNKNOWN]: {str(row["text"]).strip()}')
                for _, r in right.iterrows():
                    lines.append(f'{r["role"]}: {str(r["text"]).strip()}')
                contexts.append("\n".join(lines))
                positions.append((row["conversation_id"], t))

        labels: List[Optional[str]] = []
        if contexts:
            # Build prompts once
            prompts = []
            for ctx in contexts:
                prompts.append(
                    "In the following weight-loss coaching dialogue snippet, one line is marked as [UNKNOWN]. "
                    "Classify if that [UNKNOWN] line was spoken by the advisor (coach) or the patient. "
                    "Answer with exactly one word: advisor or patient.\n\n"
                    f"{ctx}\n\nLabel:"
                )

            # Initialize pipeline once
            llm._lazy_init()

            if _DATASETS_OK:
                # STREAMED: one pipeline call over a dataset iterator
                ds = Dataset.from_dict({"text": prompts})
                it = llm._pipe(
                    KeyDataset(ds, "text"),
                    return_full_text=False,
                    batch_size=llm.batch_size,
                )
                for out in tqdm(it, total=len(prompts), desc="Resolve UNKNOWN (LLM, streamed)"):
                    # text-generation pipeline yields list[dict] per item
                    if isinstance(out, list) and out and isinstance(out[0], dict):
                        gen = out[0].get("generated_text", "") or out[0].get("summary_text", "")
                    elif isinstance(out, dict):
                        gen = out.get("generated_text", "") or out.get("summary_text", "")
                    else:
                        gen = ""
                    low = gen.strip().lower()
                    if "advisor" in low and "patient" in low:
                        low = low.split()[0]
                    if "advisor" in low or "coach" in low:
                        labels.append("advisor")
                    elif "patient" in low:
                        labels.append("patient")
                    else:
                        labels.append(None)
            else:
                # Fallback: chunked loop (still shows tqdm, but multiple pipeline calls)
                print("[note] 'datasets' not available; falling back to chunked batching (you may see a warning).")
                bs = llm.batch_size
                for i in tqdm(range(0, len(prompts), bs), desc="Resolve UNKNOWN (LLM, batched)"):
                    chunk = prompts[i:i+bs]
                    outs = llm.infer_many(chunk)
                    for gen in outs:
                        low = (gen or "").strip().lower()
                        if "advisor" in low and "patient" in low:
                            low = low.split()[0]
                        if "advisor" in low or "coach" in low:
                            labels.append("advisor")
                        elif "patient" in low:
                            labels.append("patient")
                        else:
                            labels.append(None)

            # Apply labels back
            lut = {pos: lab for pos, lab in zip(positions, labels)}
            for r in role_rows:
                if r["role"] == "unknown":
                    lab = lut.get((r["conversation_id"], int(r["turn_index"])))
                    if lab in {"advisor","patient"}:
                        r["role"] = lab
            for r in clean_rows:
                if r["role"] == "unknown":
                    lab = lut.get((r["conversation_id"], int(r["turn_index"])))
                    if lab in {"advisor","patient"}:
                        r["role"] = lab

    # 6) Fallback: ensure no 'unknown' remains; log which ones first ---
    unknown_positions = [
        (i, r["conversation_id"], int(r["turn_index"]), r.get("text", ""))
        for i, r in enumerate(clean_rows)
        if str(r["role"]).lower() == "unknown"
    ]
    unknown_left = len(unknown_positions)

    if unknown_left > 0:
        # Print a compact list of remaining unknowns
        print(f"[warn] {unknown_left} unknown roles remain; defaulting them to 'advisor'.")
        # If there are many, show a head; otherwise show all
        MAX_SHOW = 50
        to_show = unknown_positions[:MAX_SHOW]
        for i, cid, tix, txt in to_show:
            preview = (txt[:120] + "…") if isinstance(txt, str) and len(txt) > 120 else txt
            print(f"  - clean_rows[{i}]  conv={cid}  turn_index={tix}  text={preview!r}")
        if unknown_left > MAX_SHOW:
            print(f"  … and {unknown_left - MAX_SHOW} more (not shown).")

        # (Optional) also dump a CSV for auditing
        audit_path = "unresolved_unknowns.csv"
        import pandas as pd
        pd.DataFrame(
            [{"clean_rows_index": i, "conversation_id": cid, "turn_index": tix, "text": txt}
            for (i, cid, tix, txt) in unknown_positions]
        ).to_csv(audit_path, index=False, encoding="utf-8")
        print(f"[audit] Wrote unresolved unknowns to: {audit_path}")

        # Now default them
        for i, _, _, _ in unknown_positions:
            clean_rows[i]["role"] = "advisor"


    # 7) Write outputs
    pd.DataFrame(role_rows).to_csv(out_roles, index=False, encoding="utf-8")
    pd.DataFrame(clean_rows).to_csv(out_clean, index=False, encoding="utf-8")
    print(f"[write] Wrote role-assigned CSV: {out_roles}")
    print(f"[write] Wrote cleaned CSV:      {out_clean}")

    # 8) Clean cache
    import torch
    with torch.no_grad():
        torch.cuda.empty_cache()

def parse_args():
    ap = argparse.ArgumentParser(description="Merged: role assignment + resolve unknowns + cleaning (batched LLM for unknowns only).")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--out_folder", type=str, default="./data/coaching_en/")
    ap.add_argument("--out_roles", type=str, default="smartcoaching_roles_batch.csv")
    ap.add_argument("--out_clean", type=str, default="smartcoaching_clean_batch.csv")
    ap.add_argument("--use_llm", type=str, default="false")
    ap.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--llm_batch_size", type=int, default=16)
    ap.add_argument("--window_left", type=int, default=4)
    ap.add_argument("--window_right", type=int, default=2)
    ap.add_argument("--drop_max_chars", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    use_llm = str(args.use_llm).strip().lower() in {"1","true","yes","y"}
    output_roles_path = args.out_folder + args.out_roles
    output_clean_path = args.out_folder + args.out_clean
    process_file(
        input_csv=args.input_csv,
        out_roles=output_roles_path,
        out_clean=output_clean_path,
        use_llm=use_llm,
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        llm_batch_size=args.llm_batch_size,
        window_left=args.window_left,
        window_right=args.window_right,
        drop_max_chars=args.drop_max_chars,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
