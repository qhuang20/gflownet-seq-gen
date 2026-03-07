"""Generate target RNA sequences for GFlowNet training."""

import random
import json
from pathlib import Path


def generate_targets(n_targets=100, seq_len=10, seed=42):
    """Generate diverse RNA target sequences using multiple strategies."""
    random.seed(seed)
    alphabet = ['A', 'U', 'G', 'C']
    targets = set()

    # Homopolymers
    for base in alphabet:
        targets.add(base * seq_len)

    # Dinucleotide repeats
    for b1 in alphabet:
        for b2 in alphabet:
            if b1 != b2:
                pattern = (b1 + b2) * (seq_len // 2)
                targets.add(pattern[:seq_len])

    # Trinucleotide repeats
    codons = ['AUG', 'UAG', 'UGA', 'UAA', 'GCG', 'CUA', 'AGG', 'UUU', 'AAA', 'GGG']
    for codon in codons:
        pattern = (codon * (seq_len // 3 + 1))[:seq_len]
        targets.add(pattern)

    # Palindromic sequences
    def make_palindrome(half):
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        return half + ''.join(complement[b] for b in reversed(half))

    for _ in range(10):
        half = ''.join(random.choices(alphabet, k=seq_len // 2))
        targets.add(make_palindrome(half))

    # GC-rich
    for _ in range(10):
        targets.add(''.join(random.choices(['G', 'C'], k=seq_len)))

    # AU-rich
    for _ in range(10):
        targets.add(''.join(random.choices(['A', 'U'], k=seq_len)))

    # Fill remaining with random
    while len(targets) < n_targets:
        targets.add(''.join(random.choices(alphabet, k=seq_len)))

    return sorted(list(targets))[:n_targets]


def save_targets(targets, output_path):
    """Save targets as JSON, text, and Python module."""
    output_path = Path(output_path)

    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(targets, f, indent=2)

    with open(output_path.with_suffix('.txt'), 'w') as f:
        for seq in targets:
            f.write(seq + '\n')

    with open(output_path.with_suffix('.py'), 'w') as f:
        f.write('"""Auto-generated RNA target sequences."""\n\n')
        f.write(f'# {len(targets)} target sequences of length {len(targets[0])}\n')
        f.write('TARGET_SEQUENCES = [\n')
        for seq in targets:
            f.write(f'    "{seq}",\n')
        f.write(']\n\n')
        f.write('TARGET_SEQUENCES_LIST = [list(s) for s in TARGET_SEQUENCES]\n')

    print(f"Saved {len(targets)} targets to {output_path}")


def main():
    targets = generate_targets(n_targets=100, seq_len=10, seed=42)
    print(f"Generated {len(targets)} unique target sequences (length {len(targets[0])})")

    gc_contents = [seq.count('G') + seq.count('C') for seq in targets]
    print(f"GC content: min={min(gc_contents)*10:.0f}%, max={max(gc_contents)*10:.0f}%, "
          f"mean={sum(gc_contents)/len(gc_contents)*10:.0f}%")

    output_path = Path(__file__).parent / 'targets_10bp'
    save_targets(targets, output_path)


if __name__ == '__main__':
    main()
