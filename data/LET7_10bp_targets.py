"""LET-7 10bp target sequences (positions 10-19 of 22bp, selected for max diversity)."""

import json
from collections import Counter
from pathlib import Path

INPUT_JSON = Path(__file__).parent / "LET7_22bp_targets.json"
OUTPUT_JSON = Path(__file__).parent / "LET7_10bp_targets.json"

START_POS = 9   # 0-indexed
END_POS = 19
SEQ_LEN = END_POS - START_POS


def load_sequences():
    with open(INPUT_JSON, 'r') as f:
        return json.load(f)


def truncate_sequences(seqs_dict, start=START_POS, end=END_POS):
    return {name: seq[start:end] for name, seq in seqs_dict.items()}


def generate_targets():
    seqs_22bp = load_sequences()
    print(f"Loaded {len(seqs_22bp)} 22bp sequences")

    seqs_10bp = truncate_sequences(seqs_22bp)
    unique_seqs = sorted(set(seqs_10bp.values()))

    print(f"Truncation: positions {START_POS+1}-{END_POS} -> {len(unique_seqs)} unique 10bp sequences")

    seq_counts = Counter(seqs_10bp.values())
    print(f"\nTop 10 most common:")
    for seq, count in seq_counts.most_common(10):
        print(f"  {seq}: {count} species")

    return seqs_10bp, unique_seqs


def save_outputs(seqs_10bp, unique_seqs):
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(seqs_10bp, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Targets: {len(unique_seqs)} unique / {4**SEQ_LEN:,} possible ({len(unique_seqs)/4**SEQ_LEN:.2e} density)")


# Pre-computed targets for import
TARGET_SEQUENCES_DICT = None
TARGET_SEQUENCES_LIST = None


def _load_targets():
    global TARGET_SEQUENCES_DICT, TARGET_SEQUENCES_LIST
    if TARGET_SEQUENCES_DICT is None:
        try:
            with open(OUTPUT_JSON, 'r') as f:
                TARGET_SEQUENCES_DICT = json.load(f)
            TARGET_SEQUENCES_LIST = sorted(set(TARGET_SEQUENCES_DICT.values()))
        except FileNotFoundError:
            print("Warning: LET7_10bp_targets.json not found. Run this script to generate.")
            TARGET_SEQUENCES_DICT = {}
            TARGET_SEQUENCES_LIST = []
    return TARGET_SEQUENCES_DICT, TARGET_SEQUENCES_LIST


_load_targets()

if __name__ == "__main__":
    seqs_10bp, unique_seqs = generate_targets()
    save_outputs(seqs_10bp, unique_seqs)
