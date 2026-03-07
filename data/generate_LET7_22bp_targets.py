"""Filter LET-7 miRNA sequences to 22bp for GFlowNet training."""

import json
from collections import Counter
from pathlib import Path

INPUT_FASTA = Path(__file__).parent / "LET7_family_mature_ALLspecies.fa"
OUTPUT_JSON = Path(__file__).parent / "LET7_22bp_targets.json"
OUTPUT_FASTA = Path(__file__).parent / "LET7_22bp_sequences.fa"


def parse_fasta(filepath):
    """Parse FASTA file, return dict of name -> sequence."""
    sequences = {}
    current_name = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_name = line[1:]
            elif current_name and line:
                sequences[current_name] = line

    return sequences


def main():
    all_seqs = parse_fasta(INPUT_FASTA)
    print(f"Total sequences: {len(all_seqs)}")

    lengths = Counter(len(seq) for seq in all_seqs.values())
    print(f"\nLength distribution:")
    for length, count in sorted(lengths.items()):
        print(f"  {length}bp: {count} sequences")

    seqs_22bp = {name: seq for name, seq in all_seqs.items() if len(seq) == 22}
    unique_seqs = set(seqs_22bp.values())
    print(f"\n22bp: {len(seqs_22bp)} total, {len(unique_seqs)} unique")

    seq_counts = Counter(seqs_22bp.values())
    print(f"\nTop 10 most common:")
    for seq, count in seq_counts.most_common(10):
        print(f"  {seq}: {count} species")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(seqs_22bp, f, indent=2)
    print(f"\nSaved JSON: {OUTPUT_JSON}")

    with open(OUTPUT_FASTA, 'w') as f:
        for name, seq in seqs_22bp.items():
            f.write(f">{name}\n{seq}\n")
    print(f"Saved FASTA: {OUTPUT_FASTA}")

    print(f"\nTargets: {len(unique_seqs)} unique / {4**22:,} possible ({len(unique_seqs)/4**22:.2e} density)")


if __name__ == "__main__":
    main()
