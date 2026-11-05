"""
Step 4. Build the cluster-based cross-validation splits.


Usage:
    python 4_split_data.py --setting protein --threshold 0.5
    python 4_split_data.py --setting peptide --threshold 0.5
    python 4_split_data.py --setting pair    --threshold 0.5
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.model_selection import KFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


SEEDS = {
    'protein': (1024, 42),
    'peptide': (42, 59),
    'pair': (1024, 42),
}


PAIR_FOLDS_PER_SIDE = 3


def parse_args():
    p = argparse.ArgumentParser(
        description='Build cluster-based cross-validation splits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--setting', required=True, choices=['protein', 'peptide', 'pair'],
                   help='which side the folds are drawn over')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='clustering threshold whose cluster dictionary to use')
    p.add_argument('--n-folds', type=int, default=None,
                   help='5 for protein/peptide; for pair this is the folds per side, '
                        f'default {PAIR_FOLDS_PER_SIDE}, giving its square many splits')
    p.add_argument('--cluster-dir', default=os.path.join(HERE, 'similarity_cluster_dict'),
                   help='directory holding {pep,prot}_cluster_dict_<threshold>.pkl')
    p.add_argument('--pairs-dir', default=os.path.join(REPO, 'data', 'prot_5fold_5'),
                   help='an existing split directory; the union of one fold defines the '
                        'universe of (peptide, protein) pairs to redistribute')
    p.add_argument('--pairs-fold', type=int, default=1)
    p.add_argument('--out-dir', default=None,
                   help='defaults to <repo>/data/<prot|pep|pair>_<n>fold_<threshold*10>')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='override the default seeds for this setting')
    p.add_argument('--dry-run', action='store_true',
                   help='report fold sizes and ratios without writing any files')
    return p.parse_args()


def load_pairs(pairs_dir, fold):

    pairs, seen = [], set()
    for part in ('train', 'val', 'test'):
        path = os.path.join(pairs_dir, f'{part}_data_5_fold_{fold}.pkl')
        with open(path, 'rb') as f:
            for pair in pickle.load(f):
                key = tuple(pair)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
    return pairs


def peptide_chain(pair):

    parts = pair[0].split('_')
    return parts[0] + '_' + parts[1]


def load_clusters(cluster_dir, prefix, threshold):
    path = os.path.join(cluster_dir, f'{prefix}_cluster_dict_{threshold}.pkl')
    with open(path, 'rb') as f:
        clusters = pickle.load(f)
    return clusters, np.array(list(clusters.keys()))


def cluster_folds(cluster_list, n_folds, kfold_seed, val_seed):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=kfold_seed)
    folds = []
    for train_index, test_index in kf.split(cluster_list):
        train_part, val_part = train_test_split(
            train_index, test_size=0.1, random_state=val_seed)
        folds.append((cluster_list[train_part], cluster_list[val_part],
                      cluster_list[test_index]))
    return folds


def chains_of(cluster_labels, cluster_dict):
    chains = set()
    for label in cluster_labels:
        chains.update(cluster_dict[label])
    return chains


def split_one_side(pairs, side, cluster_dict, folds):

    index = {}
    for pair in pairs:
        key = pair[1] if side == 'protein' else peptide_chain(pair)
        index.setdefault(key, []).append(pair)

    out = []
    for train_cl, val_cl, test_cl in folds:
        parts = {}
        for name, labels in (('train', train_cl), ('val', val_cl), ('test', test_cl)):
            collected = []
            for chain_id in chains_of(labels, cluster_dict):
                collected.extend(index.get(chain_id, []))
            parts[name] = collected
        out.append((parts, 0))
    return out


def split_both_sides(pairs, prot_dict, pep_dict, prot_folds, pep_folds):

    out = []
    for prot_fold in prot_folds:
        for pep_fold in pep_folds:
            prot_sets = [chains_of(labels, prot_dict) for labels in prot_fold]
            pep_sets = [chains_of(labels, pep_dict) for labels in pep_fold]
            parts = {'train': [], 'val': [], 'test': []}
            dropped = 0
            for pair in pairs:
                pep_id, prot_id = peptide_chain(pair), pair[1]
                for name, i in (('train', 0), ('val', 1), ('test', 2)):
                    if prot_id in prot_sets[i] and pep_id in pep_sets[i]:
                        parts[name].append(pair)
                        break
                else:
                    dropped += 1
            out.append((parts, dropped))
    return out


def main():
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else SEEDS[args.setting]
    prefix = {'protein': 'prot', 'peptide': 'pep', 'pair': 'pair'}[args.setting]

    pairs = load_pairs(args.pairs_dir, args.pairs_fold)
    print(f'setting           : new {args.setting}')
    print(f'pairs             : {len(pairs)}')

    if args.setting == 'pair':
        per_side = args.n_folds or PAIR_FOLDS_PER_SIDE
        n_splits = per_side ** 2
        if len(seeds) != 4:
            raise SystemExit('the pair setting needs four seeds: '
                             'protein folds, peptide folds, protein val, peptide val')
        prot_dict, prot_list = load_clusters(args.cluster_dir, 'prot', args.threshold)
        pep_dict, pep_list = load_clusters(args.cluster_dir, 'pep', args.threshold)
        print(f'clusters          : {len(prot_list)} protein, {len(pep_list)} peptide '
              f'(threshold {args.threshold})')
        print(f'grid              : {per_side} x {per_side} = {n_splits} splits')
        print(f'seeds             : prot folds {seeds[0]}, pep folds {seeds[1]}, '
              f'prot val {seeds[2]}, pep val {seeds[3]}')
        results = split_both_sides(
            pairs, prot_dict, pep_dict,
            cluster_folds(prot_list, per_side, seeds[0], seeds[2]),
            cluster_folds(pep_list, per_side, seeds[1], seeds[3]))
    else:
        n_splits = args.n_folds or 5
        if len(seeds) != 2:
            raise SystemExit('this setting needs two seeds: fold assignment, '
                             'train/validation split')
        cluster_dict, cluster_list = load_clusters(
            args.cluster_dir, prefix, args.threshold)
        print(f'clusters          : {len(cluster_list)} (threshold {args.threshold})')
        print(f'seeds             : kfold {seeds[0]}, val {seeds[1]}')
        results = split_one_side(
            pairs, args.setting, cluster_dict,
            cluster_folds(cluster_list, n_splits, seeds[0], seeds[1]))

    out_dir = args.out_dir or os.path.join(
        REPO, 'data', f'{prefix}_{n_splits}fold_{int(args.threshold * 10)}')
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
    tag = int(args.threshold * 10)
    ratios = []

    for fold, (parts, dropped) in enumerate(results, 1):
        sizes = {k: len(v) for k, v in parts.items()}
        total = sum(sizes.values())
        ratios.append([sizes[k] / max(total, 1) for k in ('train', 'val', 'test')])
        line = f'  fold {fold:2d}: ' + '  '.join(
            f'{k} {v} ({v/max(total,1):.1%})' for k, v in sizes.items())
        if args.setting == 'pair':
            line += f'   kept {total}, dropped {dropped}'
        print(line)
        if args.dry_run:
            continue
        for part, data in parts.items():
            path = os.path.join(out_dir, f'{part}_data_{tag}_fold_{fold}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    mean = np.mean(ratios, axis=0)
    print(f'\nmean ratio        : train {mean[0]:.1%}  val {mean[1]:.1%}  test {mean[2]:.1%}')
    if args.dry_run:
        print('dry run: nothing written')
    else:
        print(f'wrote {n_splits} folds to {out_dir}')


if __name__ == '__main__':
    main()
