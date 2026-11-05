# ncPPI-Bind

Dynamic Class Balancing and Site-Specific Contrastive Learning Enhance Non-Canonical Peptide–Protein Binding Site Prediction

This repository contains the source code, model configuration, cross-validation splits, and residue labels. Demo features and model checkpoints are available from Zenodo: [https://doi.org/10.5281/zenodo.23853930](https://doi.org/10.5281/zenodo.23853930).

## Installation

The environment file specifies a Linux x86_64 / CUDA 11.8 environment.

```bash
conda env create -f environment.yml
conda activate ncppi-bind
```

## Demo

Download `ncPPI-Bind_demo_200.tar.gz` and `ncPPI-Bind_checkpoints.tar.gz` from the Zenodo record, extract both archives into the repository root and run the demo:

```bash
tar -xzf ncPPI-Bind_demo_200.tar.gz
tar -xzf ncPPI-Bind_checkpoints.tar.gz
python demo.py --mode all --device cuda:0
```

| `--mode` | Steps |
|---|---|
| `test` | Evaluates the fold-1 checkpoint on the 30 demo test complexes. |
| `train` | Trains for one epoch on the demo training set and evaluates the resulting checkpoint. |
| `all` | Runs `test`, then `train`. |

Outputs are written to `demo/outputs/<run tag>/`.


## Structure files

mmCIF files are not included in this repository. Before data processing, download the mmCIF file of each PDB entry in the split files from [RCSB PDB](https://www.rcsb.org/downloads) into `data/mmcif_files/`, uncompressed and named `<pdb_id>.cif` in lower case (for example `data/mmcif_files/7z8e.cif`). 



## Cross-validation splits

The split files under `data/` were produced by the pipeline in `cluster/`. See `cluster/README.md` to rebuild them or to change the similarity threshold.


## Features for training and testing

Full training and testing require feature files for every complex in the selected split. Feature paths are set in `config.json` and are resolved relative to that file.

| Config field | File name | Contents |
|---|---|---|
| `peptide_graph_dir` | `PDBID_PEPTIDECHAIN` (no extension) | Pickle dictionary with `node_emb [N_atoms,300]`, `nodes_residue_index [N_atoms]`, `edges [E,2]`, `edge_attrs [E]` |
| `protein_graph_dir` | `PDBID_PEPTIDECHAIN_PROTEINCHAIN.pkl` | Pickle dictionary with `edge [E,2]`, `edge_attr [E]` |
| `protein_embedding_dir` | `PDBID_PROTEINCHAIN.npy` | ProtT5 residue embeddings `[L,1024]` |
| `protein_mask_dir` | `PDBID_PROTEINCHAIN.npy` | Protein mask `[L]` |

### Peptide graphs

Peptide atom embeddings are generated with [Hi-GMAE](https://github.com/LiuChuang0059/Hi-GMAE): one `<CCD_ID>.npy` file per residue type, containing a 300-dimensional embedding per atom in the residue template atom order. Place these files in `data/aa_embed`, then run:

```bash
python peptide2graph.py \
  --splits-dir data/prot_5fold_5 \
  --cif-dir data/mmcif_files \
  --ligand-dir data/ligands \
  --embed-dir data/aa_embed \
  --out-dir features/peptide_graph \
  --workers 8
```

Missing mmCIF files and CCD residue templates are downloaded from RCSB. Use `--pairs <file>` with one `PDBID_CHAIN` per line to build graphs for a custom list of peptide chains.

### Protein graphs

With the mmCIF files in `data/mmcif_files`, run:

```bash
python prot2graph.py --workers 8
```

Graphs are written to `features/protein_graph/`.

### Protein embeddings

Generate embeddings with [ProtT5-XL-half-UniRef50](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc): one `[L,1024]` array per protein chain, one row per sequence residue, without special-token rows. Save them as `features/protT5/PDBID_PROTEINCHAIN.npy`.

## Training

```bash
python train.py --split prot_5fold_5 --fold 1
python train.py --split pep_5fold_5 --fold all --run-name new_peptide
python train.py --split random_5fold_5 --fold all --run-name random
python train.py --split pair_9fold_5 --fold all --run-name both_new
torchrun --standalone --nproc_per_node=4 train.py --split prot_5fold_5 --fold all
```

For each fold, the model is trained on the training partition. After every epoch it is evaluated on the validation partition, and the checkpoint with the lowest validation loss is kept as `best.pt` (`training.selection` in `config.json`). Hyperparameters are set in `config.json`; `--epochs` and `--seed` override the configured values.

Outputs are written to `outputs/<run-name>/<split>/fold_<n>/`:

| File | Contents |
|---|---|
| `best.pt` | Selected checkpoint |
| `last.pt` | Checkpoint after the final epoch |
| `history.jsonl` | Training loss and validation metrics per epoch |
| `config.resolved.json`, `run.json` | Configuration and run settings |

Each run requires a new `--run-name`, or an empty output directory.

## Testing

```bash
python test.py --split prot_5fold_5 --fold all --subset ncaa --output outputs/test_ncaa
python test.py --split prot_5fold_5 --fold 1 --subset all --output outputs/test_all_fold1
python test.py --split prot_5fold_5 --fold 1 --checkpoint outputs/training/prot_5fold_5/fold_1/best.pt --output outputs/test_trained
```

