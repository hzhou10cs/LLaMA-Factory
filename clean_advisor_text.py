#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean advisor turns by removing filler/hedges and keeping info-dense sentences.

Input CSV:  conversation_id, turn_index, role, text
Output CSV: same columns, with advisor rows cleaned.

Usage:
  python clean_advisor_text.py \
    --input_csv source_smartcoaching_final.csv \
    --output_csv source_smartcoaching_final_clean.csv \
    --aggressive true \
    --min_words 4 \
    --min_keep_sentences 1
"""
import argparse, re
import pandas as pd
from typing import List

# --- configuration ---
FILLER_LEADS = [
    r"so", r"well", r"okay", r"alright", r"right", r"you know",
    r"kind of", r"sort of", r"like", r"um", r"uh", r"actually",
    r"basically", r"literally", r"to be honest",
]

# Allow leading quotes/brackets/dashes before the filler, e.g. “So, …  — So, …
LEAD_RE = re.compile(
    rf"""^\s*          # start + spaces
        [\"'“”‘’(\[\-]*\s*     # optional opening quotes/brackets/dash
        (?:{'|'.join([fr'(?:{w})' for w in FILLER_LEADS])})
        [\s,.\-:–—]*    # trailing punctuation after the filler
    """,
    re.IGNORECASE | re.VERBOSE
)

MID_FILLERS = [
    r"you\s+know", r"kind\s+of", r"sort\s+of", r"i\s+mean",
    r"like", r"basically", r"actually", r"literally", r"to\s+be\s+honest"
]
# Eat optional surrounding punctuation/spaces: ", you know, " → " "
MID_RE = re.compile(
    rf"""[,\s-]*\b(?:{'|'.join(MID_FILLERS)})\b[,\s-]*""",
    re.IGNORECASE | re.VERBOSE
)
# words that signal actions / specificity
ACTION_WORDS = {
    "set","aim","track","log","walk","add","do","plan","prepare","choose","swap",
    "target","increase","reduce","limit","focus","cook","schedule","split","repeat",
    "minutes","min","steps","kcal","calorie","calories","protein","grams","g","lb","pound","week","daily","per","x",
}

# quick unit normalizations
UNIT_PATTERNS = [
    (re.compile(r"\bminutes?\b", re.IGNORECASE), "min"),
    (re.compile(r"\bpounds?\b", re.IGNORECASE), "lb"),
    (re.compile(r"\bkilocalories?\b", re.IGNORECASE), "kcal"),
    (re.compile(r"\bstep(s)? per minute\b", re.IGNORECASE), "steps/min"),
]

# --- helpers ---

def sent_split(s: str) -> List[str]:
    # light sentence splitter (keeps numbers intact)
    parts = re.split(r"(?<=[.!?])\s+|\n+", s.strip())
    return [p.strip() for p in parts if p.strip()]

def normalize_units(s: str) -> str:
    out = s
    for pat, repl in UNIT_PATTERNS:
        out = pat.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def strip_leading_filler(sentence: str) -> str:
    # repeatedly strip leading quotes+filler like “So, …” or "Well - …"
    cur = sentence
    for _ in range(4):
        new = LEAD_RE.sub("", cur)
        # also trim leftover punctuation right after removal
        new = re.sub(r"^[\"'“”‘’(\[\-.,:–—\s]+", "", new)
        if new == cur:
            break
        cur = new
    return cur

def remove_mid_fillers(sentence: str) -> str:
    s = MID_RE.sub(" ", sentence)              # remove fillers plus surrounding punctuation
    s = re.sub(r"\s{2,}", " ", s).strip()      # compact spaces
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s, flags=re.IGNORECASE)  # "the the" → "the"
    # clean up stray commas/periods left behind
    s = re.sub(r"(,|\.)\s*(\1\s*)+", r"\1 ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()

def is_info_dense(sentence: str, min_words: int) -> bool:
    words = re.findall(r"\w+", sentence.lower())
    if len(words) < min_words:
        return False
    # keep if any number or unit-like token present
    if re.search(r"\b\d+(\.\d+)?\b", sentence):
        return True
    if any(tok in words for tok in ACTION_WORDS):
        return True
    return False

def clean_advisor_text(text: str, aggressive: bool, min_words: int) -> str:
    if not text or not text.strip():
        return text
    # normalize basic spacing first
    t = re.sub(r"\s+", " ", text).strip()

    # split, clean per sentence
    sents = sent_split(t)
    cleaned = []
    for s in sents:
        s1 = strip_leading_filler(s)
        s2 = remove_mid_fillers(s1)
        s3 = normalize_units(s2)
        if aggressive:
            # drop short/low-info sents unless they include numbers
            if not is_info_dense(s3, min_words):
                continue
        cleaned.append(s3)

    # de-duplicate exact sentences while preserving order
    seen = set()
    deduped = []
    for s in cleaned:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    # reduce repeated commas/periods
    out = " ".join(deduped)
    out = re.sub(r"(,|\.)\s*(\1\s*)+", r"\1 ", out)
    return out.strip()

def process_csv(input_csv: str, output_csv: str, aggressive: bool, min_words: int, min_keep_sentences: int):
    df = pd.read_csv(input_csv)
    required = {"conversation_id","turn_index","role","text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns}")

    def maybe_clean(row):
        if str(row["role"]).lower() != "advisor":
            return row["text"]
        cleaned = clean_advisor_text(str(row["text"]), aggressive=aggressive, min_words=min_words)
        # if we removed too much, fall back to original
        if not cleaned:
            return row["text"]
        # optional: enforce minimum number of sentences kept
        if min_keep_sentences > 0:
            if len(sent_split(cleaned)) < min_keep_sentences and is_info_dense(cleaned, min_words):
                # keep as-is; otherwise fallback to original
                return cleaned
        return cleaned

    df["text"] = df.apply(maybe_clean, axis=1)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[done] Wrote cleaned CSV: {output_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Defiller for advisor turns.")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, default="source_smartcoaching_final_clean.csv")
    ap.add_argument("--aggressive", type=str, default="true", help="drop low-info sentences (true/false)")
    ap.add_argument("--min_words", type=int, default=4, help="min words for a sentence to be kept when aggressive")
    ap.add_argument("--min_keep_sentences", type=int, default=1, help="min sentences to keep if info-dense")
    return ap.parse_args()

def main():
    args = parse_args()
    aggressive = str(args.aggressive).lower() in {"1","true","yes","y"}
    process_csv(args.input_csv, args.output_csv, aggressive, args.min_words, args.min_keep_sentences)

if __name__ == "__main__":
    main()
