#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-first role assignment + advisor text cleaning + trivial coach coalescing.

What it does
------------
1) Reads original CSV (robust UTF-8 re-encode).
2) Extracts utterance segments (speaker label + text + start/end).
3) Classifies EVERY utterance as advisor/patient via a small LLM using a sliding context window.
4) Anti-ventriloquism pass: flips very-short middle acks when surrounded by same-role neighbors (optional).
5) Advisor de-filler: removes fillers ("you know", "kind of", leading "So," etc.), collapses dup sentences.
6) Trivial coach coalescing:
   - If an advisor turn is trivial (ack/backchannel) AND there is a substantive advisor turn within the next 2 turns,
     merge the trivial text into that next advisor (prepend a brief ack) and DROP the trivial line.
   - Otherwise (e.g., last reply in a section), keep it to avoid Patient→Patient adjacency.
7) Writes:
   - roles CSV (audit): conversation_id, turn_index, speaker_original, role, text, mapping_method/confidence
   - clean CSV (final): conversation_id, turn_index (recomputed), role, text

Recommended models for step 3:
- google/gemma-3-4b-it (fast, strong, fits on a single GPU)
- Qwen/Qwen2.5-1.5B-Instruct (also fine)

Usage
-----
python csv_clean.py \
  --input_csv SmartCoachingCalls.csv \
  --results_col results \
  --out_folder ./data/coaching_en/ \
  --out_roles smartcoaching_roles_llm.csv \
  --out_clean smartcoaching_clean_llm.csv \
  --model_id google/gemma-3-4b-it \
  --llm_batch_size 16 \
  --window_left 4 \
  --window_right 2 \
  --anti_vent true \
  --coalesce_next_k 2
"""

import argparse, ast, csv, json, re, sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional, but recommended: GPU-efficient streaming
_DATASETS_OK = True
try:
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
except Exception:
    _DATASETS_OK = False

from tqdm.auto import tqdm

# ----------------------- Normalization helpers -----------------------

TRIVIAL_ACKS = {
    "ok","okay","okey","okey-dokey","yes","yeah","yep","yup","uh-huh","mm-hmm","mmhmm","mm hmm","mhm",
    "thanks","thank you","thx","ty","tysm","sure","right","alright","all right",
    "hmm","uh","um","mmm","huh","great","cool","nice","awesome","perfect","fine","good","got it",
    "understood","i see","sounds good","makes sense","indeed","exactly","correct"
}

ACK_RE = re.compile(
    r"^\s*(ok(?:ay)?|k|kk|sure|yes|yeah|yep|yup|uh-?huh|mm-?hmm|thanks|thank you|sounds good|alright|right|good|fine|great)\s*\.?\s*$",
    re.I
)

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_trivial_ack(text: str, max_chars:int=24) -> bool:
    t = (text or "").strip()
    if len(t) <= max_chars:
        nm = _norm(t)
        if nm in TRIVIAL_ACKS or bool(ACK_RE.match(t.strip())):
            return True
    return False

def looks_trivial_advisor(text: str) -> bool:
    """Short coach reply without verbs/numbers (low-value as a label)."""
    t = (text or "").strip()
    if len(t.split()) <= 3 and not re.search(r"\b\d|\b(plan|try|do|add|aim|goal|walk|eat|log|track|weigh|minutes?|protein|calories|kcal|plate|portion|serve|grams?)\b", t, re.I):
        return True
    return bool(ACK_RE.match(t))

# ----------------------- Segment extraction -----------------------

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
    base = None
    if isinstance(results_obj, dict):
        base = results_obj.get("segments")
        if not isinstance(base, list):
            for v in results_obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "text" in v[0]:
                    base = v; break
    elif isinstance(results_obj, list):
        base = results_obj
    if not base: return segs
    for i, item in enumerate(base):
        if not isinstance(item, dict): continue
        spk = item.get("speaker_label") or item.get("speaker") or item.get("label") or item.get("SPEAKER") or ""
        txt = item.get("text") or item.get("utterance") or ""
        if not str(txt).strip(): continue
        segs.append({"idx": i, "speaker": str(spk), "text": str(txt), "start": item.get("start"), "end": item.get("end")})
    return segs

# ----------------------- LLM wrapper -----------------------

class SmallLLM:
    def __init__(self, model_id: str, device: Optional[str], max_new_tokens: int, batch_size: int):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._pipe = None
        self._tok = None

    def _lazy_init(self):
        if self._pipe is not None: return
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tok.pad_token_id is None:
            if tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
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
        self._tok = tok
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

    def infer_stream(self, prompts: List[str]):
        self._lazy_init()
        if _DATASETS_OK:
            ds = Dataset.from_dict({"text": prompts})
            return self._pipe(KeyDataset(ds, "text"), return_full_text=False, batch_size=self.batch_size)
        else:
            for i in range(0, len(prompts), self.batch_size):
                chunk = prompts[i:i+self.batch_size]
                outs = self._pipe(chunk, return_full_text=False)
                for o in outs:
                    yield o

# ----------------------- Prompt building -----------------------

def build_prompt(window_lines: List[Tuple[str,str]], current_text: str) -> str:
    ctx = []
    for rh, tx in window_lines:
        ctx.append(f"{rh}: {tx.strip()}")
    ctx.append(f"[TARGET]: {current_text.strip()}")
    return (
        "You label roles in a two-person weight-loss coaching call.\n"
        "Roles: advisor (coach) vs patient. Consider local context carefully.\n"
        "Answer with EXACTLY one word: advisor or patient.\n\n"
        + "\n".join(ctx) + "\n\nLabel:"
    )

# ----------------------- Advisor text cleaner (merged) -----------------------

FILLER_PATTERNS = [
    (re.compile(r"\b(?:you know|i mean|kind of|sort of|like,?)\b", re.I), ""),   # remove common fillers
    (re.compile(r"^\s*(?:so|well|okay|alright|right|and|but)[,:\-]\s*", re.I), ""),  # leading discourse markers
    (re.compile(r"\s{2,}"), " "),  # squeeze spaces
]

def dedup_sentences(text: str) -> str:
    # Split into simple sentences and deduplicate adjacent repeats
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    out = []
    last = None
    for p in parts:
        if not p: continue
        p_norm = re.sub(r"\s+", " ", p).strip().lower()
        if p_norm and p_norm != last:
            out.append(p.strip())
            last = p_norm
    return " ".join(out)

def clean_advisor_text(text: str) -> str:
    t = text or ""
    for rx, repl in FILLER_PATTERNS:
        t = rx.sub(repl, t)
    t = re.sub(r"\s+", " ", t).strip()
    t = dedup_sentences(t)
    # prefer short, declarative sentences; trim trailing "so" fragments
    t = re.sub(r"\s*(?:so|and)\s*$", "", t, flags=re.I).strip()
    return t

# ----------------------- Anti-ventriloquism (optional) -----------------------

def anti_ventriloquism_fix(rows: List[Dict[str,Any]]) -> None:
    """
    If a very short ack is sandwiched between identical roles (advisor ... advisor),
    flip the middle to the opposite role. Strict condition to avoid over-flips.
    """
    for i in range(1, len(rows)-1):
        L, M, R = rows[i-1], rows[i], rows[i+1]
        if L["role"] in {"advisor","patient"} and R["role"] == L["role"] and M["role"] in {"advisor","patient"} and M["role"] != L["role"]:
            if is_trivial_ack(M["text"], max_chars=16) or len(M["text"].split()) <= 3:
                M["role"] = "patient" if L["role"] == "advisor" else "advisor"

# ----------------------- Coalesce trivial advisor replies -----------------------

def coalesce_trivial_advisors(conv_rows: List[Dict[str,Any]], lookahead_k:int=2) -> List[Dict[str,Any]]:
    """
    Merge trivial advisor acks into the NEXT substantive advisor within K turns.
    If no such advisor exists soon, keep the trivial line to avoid P→P adjacency.
    Recompute turn_index afterwards.
    """
    keep: List[Dict[str,Any]] = []
    i = 0
    n = len(conv_rows)
    while i < n:
        cur = conv_rows[i]
        if cur["role"] == "advisor" and looks_trivial_advisor(cur["text"]):
            # look ahead for next substantive advisor within K turns
            merged = False
            for j in range(1, lookahead_k+1):
                if i + j >= n: break
                nxt = conv_rows[i + j]
                if nxt["role"] == "advisor" and not looks_trivial_advisor(nxt["text"]):
                    # merge trivial ack into next advisor as a brief parenthetical prefix
                    prefix = re.sub(r"\s+", " ", cur["text"]).strip().rstrip(".")
                    if prefix:
                        nxt["text"] = f"{prefix}. {nxt['text']}".strip()
                    merged = True
                    break
                # if we encounter another patient BEFORE a substantive advisor, continue scanning (we still prefer merge)
            if merged:
                # drop the trivial advisor line
                i += 1
                continue
            else:
                # keep it (likely end of section), to avoid P→P adjacency
                keep.append(cur)
                i += 1
        else:
            keep.append(cur); i += 1

    # Re-number turn_index per conversation
    for k, row in enumerate(keep):
        row["turn_index"] = k
    return keep

# ----------------------- Main pipeline -----------------------

def process_file(
    input_csv: str,
    out_roles: str,
    out_clean: str,
    results_col: Optional[str] = None,
    model_id: str = "google/gemma-3-4b-it",
    device: Optional[str] = "auto",
    max_new_tokens: int = 6,
    llm_batch_size: int = 16,
    window_left: int = 4,
    window_right: int = 2,
    anti_vent: bool = True,
    coalesce_next_k: int = 2,
):
    # 0) robust re-encode to utf-8
    tmp_utf8 = "__reformatted_utf8.csv"
    decoded = False
    for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
        try:
            with open(input_csv, "r", newline="", encoding=enc, errors="strict") as fin, \
                 open(tmp_utf8, "w", newline="", encoding="utf-8") as fout:
                rdr = csv.reader(fin); w = csv.writer(fout)
                for row in rdr: w.writerow(row)
            print(f"[read] decoded with {enc} -> {tmp_utf8}")
            decoded = True
            break
        except UnicodeDecodeError:
            continue
    if not decoded:
        print("[read] WARNING: could not strictly re-decode; trying lax utf-8.")
        with open(input_csv, "r", newline="", encoding="utf-8", errors="replace") as fin, \
             open(tmp_utf8, "w", newline="", encoding="utf-8") as fout:
            rdr = csv.reader(fin); w = csv.writer(fout)
            for row in rdr: w.writerow(row)

    df = pd.read_csv(tmp_utf8)
    # choose columns
    path_col = next((c for c in df.columns if str(c).strip().lower() in {"relative_path","path","file","filename","conversation_id"}), df.columns[0])
    if results_col is None:
        results_col = next((c for c in df.columns if str(c).strip().lower() in {"results","json","segments"}), None)
        if results_col is None:
            raise ValueError("Could not find a RESULTS/segments/JSON column; pass --results_col <name>.")
    print(f"[cols] conversation_id='{path_col}', results_col='{results_col}'")

    # 1) collect all utterances and LLM prompts
    all_convs: Dict[str, List[Dict[str,Any]]] = {}
    prompts: List[str] = []
    positions: List[Tuple[str,int]] = []

    for ridx, row in df.iterrows():
        cid = str(row.get(path_col, f"row_{ridx}"))
        segs = extract_segments(load_results_cell(row.get(results_col)))
        if not segs:
            continue
        all_convs[cid] = segs
        for i, seg in enumerate(segs):
            # left: last window_left; right: next window_right
            left_ctx = segs[max(0, i-window_left): i]
            right_ctx = segs[i+1: i+1+window_right]
            window_lines: List[Tuple[str,str]] = []
            for s in left_ctx:
                rh = s["speaker"] if s["speaker"] else "UNKNOWN"
                window_lines.append((rh, s["text"]))
            for s in right_ctx:
                rh = s["speaker"] if s["speaker"] else "UNKNOWN"
                window_lines.append((rh, s["text"]))
            prompts.append(build_prompt(window_lines, seg["text"]))
            positions.append((cid, i))

    # 2) LLM inference (streamed)
    print(f"[llm] classifying {len(prompts)} utterances with {model_id} (batch={llm_batch_size}) ...")
    llm = SmallLLM(model_id=model_id, device=device, max_new_tokens=max_new_tokens, batch_size=llm_batch_size)
    gen_iter = llm.infer_stream(prompts)

    raw_labels: List[str] = []
    for out in tqdm(gen_iter, total=len(prompts), desc="Role classification (LLM, streamed)"):
        if isinstance(out, list) and out and isinstance(out[0], dict):
            txt = out[0].get("generated_text", "") or out[0].get("summary_text", "")
        elif isinstance(out, dict):
            txt = out.get("generated_text", "") or out.get("summary_text", "")
        else:
            txt = ""
        low = (txt or "").strip().lower()
        lab = "advisor" if "advisor" in low or "coach" in low else ("patient" if "patient" in low else "unknown")
        raw_labels.append(lab)

    label_lut = {pos: lab for pos, lab in zip(positions, raw_labels)}

    # 3) Build per-conversation rows with cleaning & fixes
    print(" Build per-conversation rows with cleaning & fixes")
    role_rows: List[Dict[str,Any]] = []
    clean_rows: List[Dict[str,Any]] = []

    for cid, segs in all_convs.items():
        conv_rows: List[Dict[str,Any]] = []
        for i, seg in enumerate(segs):
            role = label_lut.get((cid, i), "unknown")
            txt = re.sub(r"\s+", " ", seg["text"]).strip()

            rec = {
                "conversation_id": cid,
                "turn_index": i,
                "speaker_original": seg["speaker"],
                "role": role,
                "text": txt,
                "mapping_confident": role in {"advisor","patient"},
                "mapping_method": "llm",
            }
            conv_rows.append(rec)

        # anti-ventriloquism (optional but strict)
        if anti_vent:
            anti_ventriloquism_fix(conv_rows)

        # advisor de-filler
        for r in conv_rows:
            if r["role"] == "advisor":
                r["text"] = clean_advisor_text(r["text"])

        # coalesce trivial advisor replies into next substantive advisor within K turns
        conv_rows = coalesce_trivial_advisors(conv_rows, lookahead_k=coalesce_next_k)

        # ensure no 'unknown' remains (fallback: neighbor-based or default patient)
        for idx, r in enumerate(conv_rows):
            if r["role"] == "unknown":
                left = conv_rows[idx-1]["role"] if idx-1 >= 0 else None
                right = conv_rows[idx+1]["role"] if idx+1 < len(conv_rows) else None
                if left and right and left == right:
                    r["role"] = "patient" if left == "advisor" else "advisor"
                else:
                    r["role"] = "patient"

        # append to outputs
        for r in conv_rows:
            role_rows.append({
                "conversation_id": r["conversation_id"],
                "turn_index": r["turn_index"],
                "speaker_original": r["speaker_original"],
                "role": r["role"],
                "text": r["text"],
                "mapping_confident": r["mapping_confident"],
                "mapping_method": r["mapping_method"],
            })
            clean_rows.append({
                "conversation_id": r["conversation_id"],
                "turn_index": r["turn_index"],
                "role": r["role"],
                "text": r["text"],
            })

    # 4) write files
    pd.DataFrame(role_rows).to_csv(out_roles, index=False, encoding="utf-8")
    pd.DataFrame(clean_rows).to_csv(out_clean, index=False, encoding="utf-8")
    print(f"[write] roles  -> {out_roles}")
    print(f"[write] clean  -> {out_clean}")

    # 5) quick stats
    n_unk = sum(1 for r in clean_rows if r["role"] == "unknown")
    n_rows = len(clean_rows)
    n_adv = sum(1 for r in clean_rows if r["role"] == "advisor")
    n_pat = sum(1 for r in clean_rows if r["role"] == "patient")
    print(f"[stats] rows={n_rows}, advisor={n_adv}, patient={n_pat}, unknown={n_unk}")
    # sanity: check P→P adjacency introduced (should be rare)
    pp_adj = 0
    from itertools import groupby
    for cid, group in groupby(clean_rows, key=lambda x: x["conversation_id"]):
        g = list(group)
        for i in range(len(g)-1):
            if g[i]["role"] == "patient" and g[i+1]["role"] == "patient":
                pp_adj += 1
                break
    print(f"[stats] conversations with Patient→Patient adjacency (after coalesce): {pp_adj}")

def parse_args():
    ap = argparse.ArgumentParser(description="LLM-first role assignment with advisor cleaner and trivial-coach coalescing.")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--results_col", type=str, default=None, help="Name of the JSON/segments column (default: auto-detect)")
    ap.add_argument("--out_folder", type=str, default="./data/coaching_en/")
    ap.add_argument("--out_roles", type=str, default="smartcoaching_roles_llm.csv")
    ap.add_argument("--out_clean", type=str, default="smartcoaching_clean_llm.csv")
    ap.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--llm_batch_size", type=int, default=16)
    ap.add_argument("--window_left", type=int, default=4)
    ap.add_argument("--window_right", type=int, default=2)
    ap.add_argument("--anti_vent", type=str, default="true")
    ap.add_argument("--coalesce_next_k", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    out_roles = args.out_folder.rstrip("/") + "/" + args.out_roles
    out_clean = args.out_folder.rstrip("/") + "/" + args.out_clean
    process_file(
        input_csv=args.input_csv,
        out_roles=out_roles,
        out_clean=out_clean,
        results_col=args.results_col,
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        llm_batch_size=args.llm_batch_size,
        window_left=args.window_left,
        window_right=args.window_right,
        anti_vent=str(args.anti_vent).lower() in {"1","true","yes","y"},
        coalesce_next_k=int(args.coalesce_next_k),
    )

if __name__ == "__main__":
    main()
