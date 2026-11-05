"""
Step 1a. Write the peptide and protein FASTA files that the alignment step consumes.


Usage:
    python 1_build_fasta.py
"""
import argparse
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

COLUMNS = ['pdbid', 'pep_chain_id', 'pep_aa3', 'prot_id', 'prot_seq',
           'is_standard', 'pep_seq', 'is_missing', 'missing_ratio']


def parse_args():
    p = argparse.ArgumentParser(
        description='Write peptide and protein FASTA files from the dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ppi-data', default=os.path.join(REPO, 'data', 'ppi_data.pkl'),
                   help='curated dataset pickle')
    p.add_argument('--out-dir', default=os.path.join(HERE, 'fasta'),
                   help='where the FASTA files are written')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.ppi_data, 'rb') as f:
        rows = pickle.load(f)

    peptides, proteins = {}, {}
    for row in rows:
        r = dict(zip(COLUMNS, row))
        peptides.setdefault(f"{r['pdbid']}_{r['pep_chain_id']}", r['pep_seq'])
        proteins.setdefault(f"{r['pdbid']}_{r['prot_id']}", r['prot_seq'])

    os.makedirs(args.out_dir, exist_ok=True)
    for name, records in (('peptides', peptides), ('proteins', proteins)):
        path = os.path.join(args.out_dir, f'{name}.fasta')
        with open(path, 'w') as f:
            for header, seq in records.items():
                f.write(f'>{header}\n{seq}\n')
        print(f'{len(records):6d} records -> {path}')


if __name__ == '__main__':
    main()
