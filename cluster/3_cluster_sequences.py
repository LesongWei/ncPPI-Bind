"""
Step 3. Cluster peptides and proteins by sequence similarity.


Usage:
    python 3_cluster_sequences.py --fasta fasta/proteins.fasta \
                                  --sim-mat sim_mat_protein.npy \
                                  --prefix prot
"""
import argparse
import os
import pickle

import numpy as np
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description='Cluster sequences by similarity at one or more thresholds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--fasta', required=True,
                   help='FASTA file whose record order matches the similarity matrix')
    p.add_argument('--sim-mat', required=True, help='similarity matrix from step 2')
    p.add_argument('--prefix', required=True, choices=['pep', 'prot'],
                   help='names the output files, e.g. prot_cluster_dict_0.5.pkl')
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[0.3, 0.4, 0.5, 0.6],
                   help='distance cutoffs; 0.5 is the setting used in the paper')
    p.add_argument('--out-dir', default=os.path.join(HERE, 'similarity_cluster_dict'),
                   help='where the cluster dictionaries are written')
    return p.parse_args()


def read_fasta_order(path):
    names = []
    seen = set()
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip()
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def main():
    args = parse_args()
    names = read_fasta_order(args.fasta)
    sim = np.load(args.sim_mat).astype(np.float32)
    if sim.shape[0] != len(names):
        raise ValueError(
            f'similarity matrix is {sim.shape[0]}x{sim.shape[1]} but the FASTA has '
            f'{len(names)} records; they must be built from the same file')
    print(f'{len(names)} sequences, similarity matrix {sim.shape}')

    # average linkage over the condensed distance matrix
    distance = 1.0 - sim
    np.fill_diagonal(distance, 0.0)
    distance = np.maximum(distance, distance.T)          # enforce exact symmetry
    linkage = average(squareform(distance, checks=False))

    os.makedirs(args.out_dir, exist_ok=True)
    for threshold in args.thresholds:
        labels = fcluster(linkage, threshold, 'distance')
        clusters = {}
        for label, name in zip(labels, names):
            clusters.setdefault(int(label), []).append(name)
        sizes = [len(v) for v in clusters.values()]
        print(f'threshold {threshold}: {len(clusters)} clusters, '
              f'largest {max(sizes)}, singletons {sizes.count(1)}')

        path = os.path.join(args.out_dir,
                            f'{args.prefix}_cluster_dict_{threshold}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(clusters, f)
        print(f'   -> {path}')


if __name__ == '__main__':
    main()
