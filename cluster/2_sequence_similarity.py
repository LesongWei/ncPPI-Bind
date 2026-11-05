"""
Step 2. Turn raw Smith-Waterman alignment scores into a normalised similarity matrix.

Ported to Python 3 from the CAMP clustering utilities
(https://github.com/twopin/CAMP/tree/master/cluster).

pyssw.py writes four lines per aligned pair: the two sequence names, a line holding
the optimal alignment score, and a blank separator. Raw scores are read into a square
matrix indexed by the order the sequences appear in the FASTA file, then normalised as

    similarity(i, j) = SW(i, j) / sqrt( SW(i, i) * SW(j, j) )

so that a sequence has similarity 1 with itself and the matrix is comparable across
sequences of different lengths. This is the quantity the clustering step thresholds.

Usage:
    python 2_sequence_similarity.py --fasta fasta/proteins.fasta \
                                    --alignment proteins_output.txt \
                                    --out sim_mat_protein.npy
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description='Normalise Smith-Waterman scores into a similarity matrix.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--fasta', required=True,
                   help='FASTA file whose record order defines the matrix index')
    p.add_argument('--alignment', required=True,
                   help='raw pyssw.py output for that FASTA aligned against itself')
    p.add_argument('--out', required=True, help='destination .npy file')
    return p.parse_args()


def read_fasta_order(path):
    """Map each record name to its position in the file."""
    order = {}
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip()
                order.setdefault(name, len(order))
    return order


def read_alignment_scores(path, order):
    """Fill the raw score matrix from the four-line-per-pair pyssw output."""
    n = len(order)
    scores = np.zeros((n, n), dtype=np.float64)
    with open(path) as f:
        lines = f.readlines()

    missing, malformed = 0, 0
    for i in range(0, len(lines) - 2, 4):
        a = lines[i].strip().split(' ')[-1]
        b = lines[i + 1].strip().split(' ')[-1]
        if a not in order or b not in order:
            missing += 1
            continue
        try:
            score = float(int(lines[i + 2].strip().split()[1]))
        except (IndexError, ValueError):
            malformed += 1
            continue
        scores[order[a], order[b]] = score

    if missing:
        print(f'warning: {missing} pair(s) referenced names absent from the FASTA')
    if malformed:
        print(f'warning: {malformed} pair(s) had an unreadable score line')
    return scores


def normalise(scores):
    """similarity(i, j) = SW(i, j) / sqrt(SW(i, i) * SW(j, j))."""
    self_scores = np.diag(scores).astype(np.float64)
    if np.any(self_scores <= 0):
        n_bad = int((self_scores <= 0).sum())
        raise ValueError(
            f'{n_bad} sequence(s) have a non-positive self-alignment score; '
            'the alignment output is incomplete')
    denom = np.sqrt(np.outer(self_scores, self_scores))
    return scores / denom


def main():
    args = parse_args()
    order = read_fasta_order(args.fasta)
    print(f'{len(order)} sequences in {args.fasta}')

    scores = read_alignment_scores(args.alignment, order)
    asym = np.abs(scores - scores.T).max()
    if asym > 0:
        print(f'note: raw score matrix is asymmetric by at most {asym:g}; '
              'symmetrising by taking the larger of each pair')
        scores = np.maximum(scores, scores.T)

    sim = normalise(scores)
    np.fill_diagonal(sim, 1.0)
    print(f'similarity matrix {sim.shape}, '
          f'off-diagonal range [{sim[~np.eye(len(sim), dtype=bool)].min():.4f}, '
          f'{sim[~np.eye(len(sim), dtype=bool)].max():.4f}]')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.save(args.out, sim.astype(np.float32))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
