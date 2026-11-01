"""
Build atom-level peptide graphs for ncPPI-Bind.

Each peptide chain is read from its mmCIF entry and every residue is expanded into
an atom graph built from the residue's PDB Chemical Component Dictionary template.
Pre-computed atom embeddings are attached per residue type, and the residue graphs
are joined by backbone peptide bonds, inter-residue covalent bonds and disulfide
bridges. Canonical and non-canonical residues follow the same path, so a modified
residue keeps its own chemistry instead of collapsing to an unknown token.

Atomic coordinates are used only to resolve inter-residue connectivity and are not
themselves model inputs.

One pickle is written per chain, named "<pdb_id>_<chain_id>", containing:

    nodes_int_feats      integer atom features
    node_emb             pre-computed atom embeddings
    nodes_residue_index  residue index of each atom
    edges, edge_attrs    covalent bonds and their bond types
    gt_positions         atom coordinates
    gt_position_exists   whether each atom is resolved in the structure

Usage
-----
    python peptide2graph.py                        # every peptide in the bundled splits
    python peptide2graph.py --pairs ids.txt        # a custom list, one PDBID_CHAIN per line
    python peptide2graph.py --workers 16           # parallel

Missing mmCIF entries and residue templates are downloaded from RCSB on demand.
The atom-embedding directory is required and cannot be downloaded; see the README.
"""
import argparse
import json
import os
import pickle
import sys
import traceback
from functools import partial
from multiprocessing import Pool

import requests

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

RCSB_STRUCTURE = 'https://files.rcsb.org/download/{}.cif'


def parse_args():
    p = argparse.ArgumentParser(
        description='Build atom-level peptide graphs for ncPPI-Bind.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--pairs', default=None, metavar='FILE',
                   help='text file with one PDBID_CHAIN per line; '
                        'if omitted, the peptide list is taken from --splits-dir')
    p.add_argument('--splits-dir', default=os.path.join(REPO, 'data', 'prot_5fold_5'),
                   help='directory holding the cross-validation split pickles')
    p.add_argument('--fold', type=int, default=1,
                   help='fold whose train+val+test lists enumerate the full dataset')
    p.add_argument('--cif-dir', default=os.path.join(REPO, 'data', 'mmcif_files'),
                   help='mmCIF entries; missing ones are downloaded from RCSB')
    p.add_argument('--ligand-dir', default=os.path.join(REPO, 'data', 'ligands'),
                   help='residue templates (.cif and _ideal.sdf); downloaded on demand')
    p.add_argument('--embed-dir', default=os.path.join(REPO, 'data', 'aa_embed'),
                   help='pre-computed per-residue atom embeddings, one <RES>.npy per residue type')
    p.add_argument('--out-dir', default=os.path.join(REPO, 'data', 'peptide_graph'),
                   help='where the peptide graphs are written')
    p.add_argument('--workers', type=int, default=1,
                   help='worker processes; 1 runs in the main process')
    p.add_argument('--overwrite', action='store_true',
                   help='rebuild graphs that already exist')
    p.add_argument('--error-log', default=os.path.join(REPO, 'peptide2graph_errors.json'),
                   help='chains that failed, with their error messages')
    return p.parse_args()


def peptide_list_from_splits(splits_dir, fold):
    """Every peptide chain of the dataset.

    Split entries are (peptide_chain, protein_chain) pairs such as
    ("2yba_D", "2yba_A"). For a given fold the train, validation and test lists
    together cover the whole dataset, so one fold is enough to enumerate it.
    """
    pairs, seen = [], set()
    for part in ('train', 'val', 'test'):
        path = os.path.join(splits_dir, f'{part}_data_5_fold_{fold}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f'split file not found: {path}')
        with open(path, 'rb') as f:
            for peptide_id, _protein_id in pickle.load(f):
                pdb_id, chain_id = peptide_id.split('_', 1)
                if (pdb_id, chain_id) in seen:
                    continue
                seen.add((pdb_id, chain_id))
                pairs.append((pdb_id, chain_id))
    if not pairs:
        raise RuntimeError(f'no peptide chains found in {splits_dir} (fold {fold})')
    return pairs


def peptide_list_from_file(path):
    pairs, seen, skipped = [], set(), []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '_' not in line:
                skipped.append(lineno)
                continue
            pdb_id, chain_id = line.split('_', 1)
            if (pdb_id, chain_id) in seen:
                continue
            seen.add((pdb_id, chain_id))
            pairs.append((pdb_id, chain_id))
    if skipped:
        print(f'warning: skipped {len(skipped)} malformed line(s) in {path} '
              f'(expected PDBID_CHAIN): lines {skipped[:10]}')
    return pairs


def download_structure(pdb_id, dst):
    r = requests.get(RCSB_STRUCTURE.format(pdb_id), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'failed to download {pdb_id}.cif (HTTP {r.status_code})')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        f.write(r.text)


def build_one(pair, cfg):
    """Build and write the graph for one peptide chain. Returns (key, error_or_None)."""
    pdb_id, chain_id = pair
    key = f'{pdb_id}_{chain_id}'
    try:
        # imported inside the worker so each process configures its own embedding path
        from utils import amino_acid
        from utils.chain import Chain
        from openfold.data.mmcif_parsing import parse

        amino_acid.set_atom_embed_dir(cfg['embed_dir'])

        dst = os.path.join(cfg['out_dir'], key)
        if os.path.exists(dst) and not cfg['overwrite']:
            return key, None

        cif_path = os.path.join(cfg['cif_dir'], f'{pdb_id}.cif')
        if not os.path.exists(cif_path):
            download_structure(pdb_id, cif_path)

        with open(cif_path) as f:
            mmcif_string = f.read()
        mmcif = parse(file_id=pdb_id, mmcif_string=mmcif_string).mmcif_object
        if mmcif is None:
            raise ValueError(f'could not parse {pdb_id}.cif')

        structure = next((c for c in mmcif.structure.get_chains()
                          if c.id == chain_id), None)
        if structure is None:
            raise ValueError(f'chain {chain_id} not found in {pdb_id}')

        chain = Chain(
            chain_id=chain_id,
            ligand_path=cfg['ligand_dir'],
            mmcif_object=mmcif,
            mmcif_chain_to_seqres=mmcif.chain_to_seqres[chain_id],
            mmcif_seq_res_to_strcture=mmcif.seqres_to_structure[chain_id],
            mmcif_structure=structure,
        )
        graph = chain.get_graph_with_gt()

        os.makedirs(cfg['out_dir'], exist_ok=True)
        with open(dst, 'wb') as f:
            pickle.dump(graph, f)
        return key, None
    except Exception:
        return key, traceback.format_exc(limit=3)


def main():
    args = parse_args()

    if args.pairs:
        pairs = peptide_list_from_file(args.pairs)
    else:
        pairs = peptide_list_from_splits(args.splits_dir, args.fold)

    if not os.path.isdir(args.embed_dir):
        sys.exit(f'atom-embedding directory not found: {args.embed_dir}\n'
                 'See the README for how to obtain it.')

    for d in (args.cif_dir, args.ligand_dir, args.out_dir):
        os.makedirs(d, exist_ok=True)

    cfg = {
        'cif_dir': args.cif_dir,
        'ligand_dir': args.ligand_dir,
        'embed_dir': args.embed_dir,
        'out_dir': args.out_dir,
        'overwrite': args.overwrite,
    }

    print(f'peptide chains to process : {len(pairs)}')
    print(f'output directory          : {args.out_dir}')
    print(f'atom embeddings           : {args.embed_dir}')
    print(f'workers                   : {args.workers}', flush=True)

    worker = partial(build_one, cfg=cfg)
    errors = {}
    done = 0

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for key, err in pool.imap_unordered(worker, pairs, chunksize=8):
                done += 1
                if err:
                    errors[key] = err
                if done % 200 == 0:
                    print(f'  {done}/{len(pairs)}  failed: {len(errors)}', flush=True)
    else:
        for pair in pairs:
            key, err = worker(pair)
            done += 1
            if err:
                errors[key] = err
            if done % 200 == 0:
                print(f'  {done}/{len(pairs)}  failed: {len(errors)}', flush=True)

    print(f'\nbuilt {done - len(errors)}/{len(pairs)} peptide graphs')
    if errors:
        with open(args.error_log, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f'{len(errors)} chains failed; details in {args.error_log}')
        for key in list(errors)[:5]:
            print(f'  {key}: {errors[key].strip().splitlines()[-1]}')


if __name__ == '__main__':
    main()
