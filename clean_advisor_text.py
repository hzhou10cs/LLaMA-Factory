#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean advisor turns by removing filler/hedges and keeping info-dense sentences.

Input CSV:  conversation_id, turn_index, role, text
Output CSV: same columns, with advisor rows cleaned.

Example run (aggressive mode):
  python clean_advisor_text.py \
    --input_csv source_smartcoaching_final.csv \
    --output_csv source_smartcoaching_final_clean.csv \
    --aggressive true \
    --min_words 4
"""
import argparse
import re
import pandas as pd
from typing import List

# ----- Config -----
FILLER_LEADS = [
    r"so", r"well", r"okay", r"alright", r"right", r"you know",
    r"kind of", r"sort of", r"like", r"um", r"uh", r"actually",
    r"basically", r"literally", r"to be honest",
]

# Allow leading quotes/brackets/dashes before the filler, e.g. “So, …  — So, …
LEAD_RE = re.compile(
    rf"""^\s*                         # start + spaces
        [\"'“”‘’(\[\-]*\s*            # optional opening quotes/brackets/dash
        (?:{'|'.join([fr'(?:{w})' for w in FILLER_LEADS])})
        [\s,.\-:–—]*                  # trailing punctuation after the filler
    """,
    re.IGNORECASE | re.VERBOSE
)

MID_FILLERS = [
    r"you\s+know", r"kind\s+of", r"sort\s+of", r"i\s+mean",r"um",
    r"like", r"basically", r"actually", r"literally", r"to\s+be\s+honest"
]
# Eat optional surrounding punctuation/spaces: ", you know, " -> " "
MID_RE = re.compile(
    rf"""[,\s-]*\b(?:{'|'.join(MID_FILLERS)})\b[,\s-]*""",
    re.IGNORECASE | re.VERBOSE
)

UNIT_PATTERNS = [
    (re.compile(r"\bminutes?\b", re.IGNORECASE), "min"),
    (re.compile(r"\bpounds?\b", re.IGNORECASE), "lb"),
    (re.compile(r"\bkilocalories?\b", re.IGNORECASE), "kcal"),
    (re.compile(r"\bstep(s)? per minute\b", re.IGNORECASE), "steps/min"),
]

ACTION_WORDS = {
    "set","aim","track","log","walk","add","do","plan","prepare","choose","swap",
    "target","increase","reduce","limit","focus","cook","schedule","split","repeat",
    "minutes","min","steps","kcal","calorie","calories","protein","grams","g","lb","pound","week","daily","per","x",
}

# ----- Helpers -----
def sent_split(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", s.strip())
    return [p.strip() for p in parts if p.strip()]

def normalize_units(s: str) -> str:
    out = s
    for pat, repl in UNIT_PATTERNS:
        out = pat.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def strip_leading_filler(sentence: str) -> str:
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
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s, flags=re.IGNORECASE)  # "the the" -> "the"
    # clean up stray commas/periods left behind
    s = re.sub(r"(,|\.)\s*(\1\s*)+", r"\1 ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()

def is_info_dense(sentence: str, min_words: int) -> bool:
    words = re.findall(r"\w+", sentence.lower())
    if len(words) < min_words:
        return False
    # keep if any number present
    if re.search(r"\b\d+(\.\d+)?\b", sentence):
        return True
    # keep if contains action-ish tokens/units
    if any(tok in words for tok in ACTION_WORDS):
        return True
    return False

def clean_advisor_text(text: str, aggressive: bool=True, min_words: int=4) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    t = re.sub(r"\s+", " ", text).strip()
    sents = sent_split(t)
    cleaned = []
    for s in sents:
        s1 = strip_leading_filler(s)
        s2 = remove_mid_fillers(s1)
        s3 = normalize_units(s2)
        if aggressive and not is_info_dense(s3, min_words):
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
    out = " ".join(deduped)
    out = re.sub(r"(,|\.)\s*(\1\s*)+", r"\1 ", out).strip()
    return out

# ----- Main -----
def process_csv(input_csv: str, output_csv: str, aggressive: bool, min_words: int):
    df = pd.read_csv(input_csv)
    required = {"conversation_id","turn_index","role","text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_csv}. Found: {df.columns}")

    df["text"] = df["text"].astype(str)
    is_adv = df["role"].str.lower() == "advisor"
    df.loc[is_adv, "text"] = df.loc[is_adv, "text"].apply(
        lambda t: clean_advisor_text(t, aggressive=aggressive, min_words=min_words)
    )

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[done] Wrote cleaned CSV: {output_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Defiller for advisor turns.")
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--out_folder", type=str, default="./data/coaching_en/")
    ap.add_argument("--output_csv", type=str, default="smartcoaching_final.csv")
    ap.add_argument("--aggressive", type=str, default="false", help="drop low-info sentences (true/false)")
    ap.add_argument("--min_words", type=int, default=4, help="min words for a sentence to be kept when aggressive")
    return ap.parse_args()

def main():
    args = parse_args()
    aggressive = str(args.aggressive).lower() in {"1","true","yes","y"}
    output_path = args.out_folder + args.output_csv
    process_csv(args.input_csv, output_path, aggressive, args.min_words)

if __name__ == "__main__":
    main()
