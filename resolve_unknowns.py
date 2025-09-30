#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve unknown roles in Smart Coaching transcripts.

Steps:
1) Remove trivial one-word/short acknowledgments that are labeled as "unknown".
2) For the remaining "unknown", infer role (advisor/patient) using context and a small LLM (batched).
3) Output a final CSV with no "unknown" roles.

Input schema (from source_smartcoaching_clean.csv):
    conversation_id, turn_index, role, text

Output schema (source_smartcoaching_final.csv):
    conversation_id, turn_index, role, text

Usage:
    python resolve_unknowns.py \
        --input_csv source_smartcoaching_clean.csv \
        --output_csv source_smartcoaching_final.csv \
        --use_llm true \
        --model_id google/gemma-2-2b-it \
        --llm_batch_size 16 \
        --window_left 4 --window_right 2 \
        --drop_max_chars 12 \
        --verbose

Notes:
- If --use_llm is false, the script will try simple context rules first. Any unresolved remainers
  will be assigned using a nearest-role heuristic (previous known role, else next known role).
- For decoder-only models (LLaMA-family), we ensure pad_token exists and set left padding for batching.
"""
import argparse, re, json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# ---------------- Trivial acknowledgments to DROP (only when role == 'unknown') ----------------
TRIVIAL_ACKS = {
    "ok","okay","okey","okey-dokey",
    "yes","yeah","yep","yup","uh-huh","mm-hmm","mmhmm","mm hmm","mhm",
    "thanks","thank you","thx","ty","tysm",
    "sure","right","alright","all right",
    "hmm","uh","um","mmm","huh",
    "great","cool","nice","awesome","perfect","fine","good","got it","understood",
    "i see","sounds good","makes sense","indeed","exactly","correct"
}

def normalize_text_for_match(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s']", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_trivial_unknown(text: str, drop_max_chars: int) -> bool:
    t = text.strip()
    if len(t) <= drop_max_chars:
        nm = normalize_text_for_match(t)
        if nm in TRIVIAL_ACKS or nm.replace("'", "") in TRIVIAL_ACKS:
            return True
        # very short (<= 2 words) generic acks
        if len(nm.split()) <= 2 and nm in {"ok","okay","yes","yeah","yep","thanks","thank you","sure","right","alright"}:
            return True
    return False

# ---------------- Small LLM (batched) ----------------
class SmallLLM:
    def __init__(self, model_id: str, device: Optional[str], batch_size: int, max_new_tokens: int = 48):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _lazy_init(self):
        if self._pipe is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

            # Ensure pad token & left padding for decoder-only models
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
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

    def _normalize_generated(self, outs: Any) -> List[str]:
        """
        HF pipeline may return:
          - list[dict] (single input) or
          - list[list[dict]] (batched)
        Normalize to list[str].
        """
        res: List[str] = []
        if isinstance(outs, list):
            if not outs:
                return res
            if isinstance(outs[0], dict):
                for o in outs:
                    res.append(o.get("generated_text", o.get("summary_text", "")))
            elif isinstance(outs[0], list):
                for sub in outs:
                    if sub and isinstance(sub[0], dict):
                        res.append(sub[0].get("generated_text", sub[0].get("summary_text", "")))
                    else:
                        res.append("")
            else:
                for _ in outs:
                    res.append("")
        else:
            res.append("")
        return res

    def infer_many(self, prompts: List[str]) -> List[str]:
        self._lazy_init()
        outs = self._pipe(prompts, return_full_text=False)
        return self._normalize_generated(outs)

    def classify_unknowns(self, contexts: List[str]) -> List[Optional[str]]:
        prompts = []
        for ctx in contexts:
            p = (
                "In the following weight-loss coaching dialogue snippet, one line is marked as [UNKNOWN]. "
                "Classify if that [UNKNOWN] line was spoken by the advisor (coach) or the patient. "
                "Answer with exactly one word: advisor or patient.\n\n"
                f"{ctx}\n\nLabel:"
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

# ---------------- Main logic ----------------
def resolve_unknowns(
    input_csv: str,
    output_csv: str,
    use_llm: bool,
    model_id: str,
    device: Optional[str],
    batch_size: int,
    window_left: int,
    window_right: int,
    drop_max_chars: int,
    verbose: bool,
):
    df = pd.read_csv(input_csv)
    required_cols = {"conversation_id","turn_index","role","text"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns}")

    # 1) Drop trivial unknowns
    mask_unknown = (df["role"].astype(str).str.lower() == "unknown")
    trivial_idx = df[mask_unknown].index[df[mask_unknown]["text"].fillna("").map(lambda x: is_trivial_unknown(str(x), drop_max_chars))]
    if verbose:
        print(f"[info] Dropping {len(trivial_idx)} trivial unknown lines.")
    df = df.drop(index=trivial_idx).reset_index(drop=True)

    # 2) Resolve remaining unknowns using context
    mask_unknown = (df["role"].astype(str).str.lower() == "unknown")
    if mask_unknown.sum() == 0:
        if verbose:
            print("[info] No remaining unknown lines.")
        df.to_csv(output_csv, index=False, encoding="utf-8")
        return

    # Build context windows
    df = df.sort_values(["conversation_id","turn_index"]).reset_index(drop=True)
    contexts: List[str] = []
    positions: List[Tuple[str,int]] = []
    grouped = df.groupby("conversation_id", sort=False)

    for cid, g in grouped:
        g = g.sort_values("turn_index")
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
            positions.append((cid, t))

    if use_llm:
        llm = SmallLLM(model_id=model_id, device=device, batch_size=batch_size)
        labels = llm.classify_unknowns(contexts)
    else:
        # Simple rule-based fallback: use nearest known role (prefer previous)
        labels = []
        for (cid, t), ctx in zip(positions, contexts):
            g = grouped.get_group(cid).sort_values("turn_index")
            prev = g[g["turn_index"] < t][::-1]
            next_ = g[g["turn_index"] > t]
            lab = None
            for _, r in prev.iterrows():
                rr = str(r["role"]).lower()
                if rr in {"advisor","patient"}:
                    lab = rr
                    break
            if lab is None:
                for _, r in next_.iterrows():
                    rr = str(r["role"]).lower()
                    if rr in {"advisor","patient"}:
                        lab = rr
                        break
            labels.append(lab or "advisor")  # default advisor

    # Apply labels
    assign = {pos: lab for pos, lab in zip(positions, labels)}
    roles = []
    for _, row in df.iterrows():
        key = (row["conversation_id"], int(row["turn_index"]))
        if key in assign:
            roles.append(assign[key])
        else:
            roles.append(row["role"])
    df["role"] = roles

    # Safety check: ensure no unknown remains
    if (df["role"].astype(str).str.lower() == "unknown").any():
        print("[warn] Some unknown labels remain; converting them to 'advisor' by default.")
        df.loc[df["role"].astype(str).str.lower() == "unknown", "role"] = "advisor"

    df.to_csv(output_csv, index=False, encoding="utf-8")
    if verbose:
        print(f"[done] Wrote final CSV without unknowns: {output_csv}")
        print(f"[stats] Final counts: {df['role'].value_counts().to_dict()}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, default="source_smartcoaching_final.csv")
    ap.add_argument("--use_llm", type=str, default="true")
    ap.add_argument("--model_id", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--llm_batch_size", type=int, default=16)
    ap.add_argument("--window_left", type=int, default=4)
    ap.add_argument("--window_right", type=int, default=2)
    ap.add_argument("--drop_max_chars", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    use_llm = str(args.use_llm).strip().lower() in {"1","true","yes","y"}
    resolve_unknowns(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        use_llm=use_llm,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.llm_batch_size,
        window_left=args.window_left,
        window_right=args.window_right,
        drop_max_chars=args.drop_max_chars,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
