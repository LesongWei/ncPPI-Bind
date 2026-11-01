"""Generate the protein-side evaluation masks.

A receptor residue is evaluated (mask 1) when it is labelled (label != -1), is one
of the 20 standard amino acids and has all three backbone atoms N, CA and C. This
is the rule of the original feature pipeline.

Masks are indexed along the mmCIF SEQRES sequence, the same coordinate as the
residue labels, the protein graphs and the ProtT5 embeddings. The OpenFold mmCIF
parser maps each SEQRES position to its resolved residue, so waters, ligands and
unresolved positions cannot shift the index.
"""
import argparse
import json
import pickle
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from Bio.PDB import Polypeptide

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_targets(data_dir, labels):
    targets = {}
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for split_file in sorted(split_dir.glob("*_data_5_fold_*.pkl")):
            with split_file.open("rb") as handle:
                pairs = pickle.load(handle)
            for peptide_id, protein_id in pairs:
                pdb_id, peptide_chain = peptide_id.split("_", 1)
                protein_chain = protein_id.split("_", 1)[1]
                label_key = f"{pdb_id}_{peptide_chain}_{protein_chain}"
                if label_key in labels:
                    targets.setdefault(
                        protein_id, (pdb_id, protein_chain, label_key)
                    )
    return targets


def make_mask(structure_file, chain_id, labels):
    from openfold.data.mmcif_parsing import parse

    mmcif = parse(
        file_id=structure_file.stem, mmcif_string=structure_file.read_text()
    ).mmcif_object
    if mmcif is None:
        raise ValueError(f"{structure_file.name}: mmCIF parser returned no structure")
    sequence = mmcif.chain_to_seqres[chain_id]
    mapping = mmcif.seqres_to_structure[chain_id]

    mask = (np.asarray(labels) != -1).astype(np.uint8)
    if len(mask) != len(sequence):
        raise ValueError(
            f"{structure_file.name} chain {chain_id}: {len(mask)} labels, "
            f"but {len(sequence)} SEQRES residues"
        )

    for index in range(len(sequence)):
        position = mapping[index]
        if position.is_missing:
            mask[index] = 0
            continue
        residue = mmcif.structure[chain_id][(
            position.hetflag,
            position.position.residue_number,
            position.position.insertion_code,
        )]
        has_backbone = all(residue.has_id(atom) for atom in ("N", "CA", "C"))
        is_standard = Polypeptide.is_aa(residue.get_resname(), standard=True)
        if not is_standard or not has_backbone:
            mask[index] = 0
    return mask


def build_one(target, labels, structure_dir, output_dir):
    protein_id, (pdb_id, chain_id, label_key) = target
    try:
        structure_file = structure_dir / f"{pdb_id}.cif"
        if not structure_file.is_file():
            raise FileNotFoundError(f"No mmCIF structure found for {pdb_id}")
        mask = make_mask(structure_file, chain_id, labels[label_key])
        np.save(output_dir / f"{protein_id}.npy", mask)
        return protein_id, None
    except Exception as error:
        return protein_id, f"{type(error).__name__}: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--structure-dir", type=Path, default=ROOT / "data/mmcif_files")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/protein_sites_label.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/protein_masks")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    labels = json.loads(args.labels.read_text())
    targets = load_targets(args.data_dir, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        (protein_id, target) for protein_id, target in sorted(targets.items())
        if args.overwrite or not (args.output_dir / f"{protein_id}.npy").exists()
    ]
    skipped = len(targets) - len(todo)
    worker = partial(
        build_one,
        labels={target[2]: labels[target[2]] for _, target in todo},
        structure_dir=args.structure_dir,
        output_dir=args.output_dir,
    )
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap(worker, todo, chunksize=16))
    else:
        results = [worker(item) for item in todo]

    errors = [(protein_id, error) for protein_id, error in results if error]
    print(f"Created {len(todo) - len(errors)} masks; skipped {skipped} existing files; "
          f"{len(errors)} failed.")
    for protein_id, error in errors[:20]:
        print(f"  {protein_id}: {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
