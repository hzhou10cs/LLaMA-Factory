#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Role assignment and cleaning for Smart Coaching call transcripts.

This version includes:
 - Two-pass, batched LLM usage
 - Summary report
 - FIX: robust handling of Hugging Face pipeline batched outputs
"""
import argparse
import ast
import json
import re
import csv
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

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
    segs = []
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

def count_matches(patterns: List[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))

def heuristic_role_scores(segments: List[Dict[str, Any]], k_first:int=12) -> Dict[str, Dict[str, int]]:
    scores: Dict[str, Dict[str, int]] = {}
    for seg in segments[:max(k_first, 1)]:
        spk = seg["speaker"]
        txt = seg["text"]
        d = scores.setdefault(spk, {"advisor": 0, "patient": 0})
        d["advisor"] += count_matches(ADVISOR_PATTERNS, txt)
        d["patient"] += count_matches(PATIENT_PATTERNS, txt)
    return scores

def decide_mapping_by_heuristics(segments: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Dict[str, Dict[str, int]], bool]:
    scores = heuristic_role_scores(segments, k_first=14)
    counts = {}
    for s in segments[:40]:
        sp = (s["speaker"] or "").strip()
        if not sp or sp.upper() == "UNKNOWN":
            continue
        counts[sp] = counts.get(sp, 0) + 1
    candidates = [sp for sp, _ in sorted(counts.items(), key=lambda x: -x[1])][:2]
    if len(candidates) < 2:
        return None, None, scores, False

    def adv_score(spk):
        sc = scores.get(spk, {"advisor":0,"patient":0})
        return sc["advisor"] - sc["patient"]

    ranked = sorted(candidates, key=lambda s: adv_score(s), reverse=True)
    adv_spk, pat_spk = ranked[0], ranked[1]
    raw_adv = scores.get(adv_spk, {}).get("advisor", 0)
    gap = adv_score(adv_spk) - adv_score(pat_spk)
    confident = (gap >= 2) or (raw_adv >= 2)
    return adv_spk, pat_spk, scores, confident

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

            # --- add these lines before building the pipeline ---
            # If tokenizer has no PAD, use EOS as PAD (common for decoder-only LMs like LLaMA)
            if tok.pad_token_id is None:
                if tok.eos_token_id is not None:
                    tok.pad_token = tok.eos_token
                    tok.pad_token_id = tok.eos_token_id
                else:
                    # last resort: add an explicit pad token and resize model embeddings
                    tok.add_special_tokens({"pad_token": "<|pad|>"})

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto" if self.device in (None, "auto") else None,
            )

            # If we added a new token above, make sure embeddings match
            if model.config.pad_token_id is None and tok.pad_token_id is not None:
                model.config.pad_token_id = tok.pad_token_id
            if getattr(model, "resize_token_embeddings", None) and len(tok) != model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(tok))

            # Decoder-only models generate best with left padding
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
        """
        HF pipeline can return:
          - list[dict] (single input)
          - list[list[dict]] (batched inputs)
        This normalizes to list[str] of generated_text.
        """
        result: List[str] = []
        if isinstance(outs, list):
            if not outs:
                return result
            if isinstance(outs[0], dict):
                # shape: [ {generated_text: ...}, ... ]
                for o in outs:
                    result.append(o.get("generated_text", o.get("summary_text", "")))
            elif isinstance(outs[0], list):
                # shape: [ [ {generated_text: ...} ], [ {..} ], ... ]
                for sub in outs:
                    if sub and isinstance(sub[0], dict):
                        result.append(sub[0].get("generated_text", sub[0].get("summary_text", "")))
                    else:
                        result.append("")
            else:
                # unexpected shape
                for _ in outs:
                    result.append("")
        else:
            result.append("")
        return result

    def infer_many(self, prompts: List[str]) -> List[str]:
        self._lazy_init()
        outs = self._pipe(prompts, return_full_text=False)
        return self._normalize_generated(outs)

    # ---- Batched helpers ----
    def map_speakers_many(self, previews: List[str]) -> List[Optional[Tuple[str,str]]]:
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

    def classify_many(self, utterances: List[str]) -> List[Optional[str]]:
        prompts = []
        for u in utterances:
            p = (
                "Classify the following single utterance from a weight-loss coaching call as spoken by the 'advisor' or the 'patient'. "
                "Reply with exactly one word: advisor or patient.\n\n"
                f"Utterance: {u}\n\nLabel:"
            )
            prompts.append(p)
        raw = self.infer_many(prompts)
        labels: List[Optional[str]] = []
        for r in raw:
            low = r.strip().lower()
            if "advisor" in low and "patient" in low:
                low = low.split()[0]
            if "advisor" in low:
                labels.append("advisor")
            elif "patient" in low:
                labels.append("patient")
            elif "coach" in low:
                labels.append("advisor")
            else:
                labels.append(None)
        return labels

def looks_like_summary(text: str) -> bool:
    txt = text.strip().lower()
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
    t = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", text)
    t = re.sub(r"\(\d{1,2}:\d{2}(?::\d{2})?\s*-\s*\d{1,2}:\d{2}(?::\d{2})?\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def process_file(
    input_csv: str,
    out_roles: str,
    out_clean: str,
    use_llm: bool = False,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    llm_batch_size: int = 1,
    verbose: bool = True,
):  
    # Reformat the source_file to line format
    reformatted_file = 'temp_reformatted.csv'
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            with open(input_csv, 'r', newline='', encoding=enc, errors='strict') as f_in, \
                open(reformatted_file, 'w', newline='', encoding='utf-8') as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                for row in reader:
                    writer.writerow(row)
            print(f"✅ Sample exported to {reformatted_file} (read with encoding={enc}, wrote as UTF-8).")
            break
        except UnicodeDecodeError as e:
            print(f"Decode failed with {enc}: {e}")
        else:
            raise RuntimeError("Could not decode the file with common encodings.")

    # Check the path and results column
    df = pd.read_csv(reformatted_file)
    path_col = next((c for c in df.columns if str(c).strip().lower() in {"relative_path", "path", "file", "filename"}), df.columns[0])
    results_col = next((c for c in df.columns if str(c).strip().lower() in {"results", "json", "segments"}), None)
    if results_col is None:
        raise ValueError("Could not locate a RESULTS/segments/JSON column in the CSV. Found: " + ", ".join(df.columns))

    llm = SmallLLM(model_id=model_id, device=device, max_new_tokens=max_new_tokens, batch_size=llm_batch_size) if use_llm else None

    # First pass
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
        adv_spk, pat_spk, scores, confident = decide_mapping_by_heuristics(segments)
        mapping[conv_id] = {"adv_spk": adv_spk, "pat_spk": pat_spk, "confident": bool(confident), "method": "heuristic"}
        if use_llm and not confident:
            need_llm_mapping.append(conv_id)

    if use_llm and need_llm_mapping:
        previews, ids = [], []
        for cid in need_llm_mapping:
            segs = all_convs[cid][:14]
            preview = "\n".join([f'{s["speaker"]}: {s["text"]}' for s in segs])
            previews.append(preview)
            ids.append(cid)
        results = llm.map_speakers_many(previews)
        for cid, res in zip(ids, results):
            if res is not None:
                adv_spk, pat_spk = res
                mapping[cid] = {"adv_spk": adv_spk, "pat_spk": pat_spk, "confident": True, "method": "llm"}

    unknown_items = []

    role_rows, clean_rows = [], []
    convo_stats = {}
    mapping_method_counts = {"heuristic": 0, "llm": 0, "fallback": 0}
    total_lines = 0
    role_counts = {"advisor": 0, "patient": 0, "unknown": 0}
    unknown_original_total = 0

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

        mapping_method_counts[method] += 1

        cs = {
            "lines": len(segments),
            "mapping_confident": bool(confident),
            "mapping_method": method,
            "unknown_original": 0,
            "unknown_resolved_by_llm": 0,
            "role_counts": {"advisor": 0, "patient": 0, "unknown": 0},
        }

        for t, seg in enumerate(segments):
            spk = (seg["speaker"] or "").strip()
            text = seg["text"]
            start = seg.get("start")
            end = seg.get("end")

            if not spk or spk.upper() == "UNKNOWN":
                cs["unknown_original"] += 1
                unknown_original_total += 1
                if use_llm:
                    unknown_items.append((conv_id, t, text))
                    role = "unknown"
                else:
                    adv_hits = count_matches(ADVISOR_PATTERNS, text)
                    pat_hits = count_matches(PATIENT_PATTERNS, text)
                    role = "advisor" if adv_hits >= max(1, pat_hits + 1) else ("patient" if pat_hits > 0 else "unknown")
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

            total_lines += 1
            role_counts[role] = role_counts.get(role, 0) + 1
            cs["role_counts"][role] = cs["role_counts"].get(role, 0) + 1

        convo_stats[conv_id] = cs

    unknown_resolved_by_llm = 0
    if use_llm and unknown_items:
        utterances = [u for (_, _, u) in unknown_items]
        labels = llm.classify_many(utterances)
        lut = {(cid, t): lab for (cid, t, _), lab in zip(unknown_items, labels)}
        for row in role_rows:
            if row["role"] == "unknown":
                lab = lut.get((row["conversation_id"], row["turn_index"]))
                if lab in {"advisor", "patient"}:
                    row["role"] = lab
                    unknown_resolved_by_llm += 1
        for row in clean_rows:
            if row["role"] == "unknown":
                lab = lut.get((row["conversation_id"], row["turn_index"]))
                if lab in {"advisor", "patient"}:
                    row["role"] = lab

        role_counts = {"advisor": 0, "patient": 0, "unknown": 0}
        for r in role_rows:
            role_counts[r["role"]] += 1

        from collections import defaultdict
        per_conv_resolved = defaultdict(int)
        for (cid, t, _), lab in zip(unknown_items, labels):
            if lab in {"advisor","patient"}:
                per_conv_resolved[cid] += 1
        for cid, cs in convo_stats.items():
            cs["unknown_resolved_by_llm"] = per_conv_resolved.get(cid, 0)

    pd.DataFrame(role_rows).to_csv(out_roles, index=False, encoding="utf-8")
    pd.DataFrame(clean_rows).to_csv(out_clean, index=False, encoding="utf-8")

    conv_total = len(convo_stats)
    confident_convs = sum(1 for v in convo_stats.values() if v["mapping_confident"])
    heuristic_mapped_convs = sum(1 for v in convo_stats.values() if v["mapping_method"]=="heuristic")
    llm_mapped_convs = sum(1 for v in convo_stats.values() if v["mapping_method"]=="llm")
    fallback_mapped_convs = sum(1 for v in convo_stats.values() if v["mapping_method"]=="fallback")

    remaining_unknown = role_counts.get("unknown", 0)
    pct_unknown = (remaining_unknown / total_lines * 100.0) if total_lines else 0.0
    pct_unknown_resolved = (unknown_resolved_by_llm / unknown_original_total * 100.0) if unknown_original_total else 0.0

    print("\\n=== Role Assignment Report ===")
    print(f"Conversations processed: {conv_total}")
    print(f" - Confident mappings: {confident_convs}")
    print(f" - Mapping methods: heuristic={heuristic_mapped_convs}, llm={llm_mapped_convs}, fallback={fallback_mapped_convs}")
    print(f"Total lines: {total_lines}  | advisor={role_counts.get('advisor',0)}, patient={role_counts.get('patient',0)}, unknown={remaining_unknown}")
    print(f"Original UNKNOWN-speaker lines: {unknown_original_total}")
    if use_llm:
        print(f"Resolved UNKNOWN lines via LLM (batched): {unknown_resolved_by_llm}  ({pct_unknown_resolved:.1f}%)")
    else:
        print("Resolved UNKNOWN lines via LLM: 0  (0.0%)  [LLM disabled]")
    print(f"Remaining UNKNOWN (all lines): {remaining_unknown}  ({pct_unknown:.1f}% of all lines)")
    print(f"Wrote role-assigned CSV: {out_roles}")
    print(f"Wrote cleaned CSV:      {out_clean}")

def parse_args():
    ap = argparse.ArgumentParser(description="Assign advisor/patient roles and clean Smart Coaching transcripts CSV (batched).")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--out_roles", type=str, default="source_smartcoaching_roleassign.csv")
    ap.add_argument("--out_clean", type=str, default="source_smartcoaching_clean.csv")
    ap.add_argument("--use_llm", type=str, default="false")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--llm_batch_size", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    use_llm = str(args.use_llm).strip().lower() in {"1","true","yes","y"}
    process_file(args.input_csv, args.out_roles, args.out_clean, use_llm,
                 args.model_id, args.device, args.max_new_tokens, args.llm_batch_size, args.verbose)

if __name__ == "__main__":
    main()