"""Dataset loader for peptide-protein binding-site prediction."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pairs(path):
    # Pairs are stored as tuples in most split files but as lists in random_5fold_5.
    with open(path, "rb") as f:
        return [tuple(pair) for pair in pickle.load(f)]


def protein_key(pair):
    pdb, peptide_chain = pair[0].split("_", 1)
    protein_chain = pair[1].split("_", 1)[1]
    return f"{pdb}_{peptide_chain}_{protein_chain}"


class SiteDataset(Dataset):
    def __init__(self, pairs, cfg):
        self.pairs = list(pairs)
        self.features = cfg["features"]
        self.peptide_labels = json.loads(
            Path(cfg["data"]["peptide_labels"]).read_text()
        )
        self.protein_labels = json.loads(
            Path(cfg["data"]["protein_labels"]).read_text()
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        pep_id, prot_id = pair
        prot_key = protein_key(pair)
        f = self.features

        with open(Path(f["peptide_graph_dir"]) / pep_id, "rb") as handle:
            peptide_graph = pickle.load(handle)
        with open(
            Path(f["protein_graph_dir"]) / f"{prot_key}.pkl", "rb"
        ) as handle:
            protein_graph = pickle.load(handle)

        peptide_embedding = torch.tensor(
            peptide_graph["node_emb"], dtype=torch.float32
        )
        peptide_index = torch.tensor(
            peptide_graph["nodes_residue_index"], dtype=torch.long
        )
        peptide_edges = torch.tensor(peptide_graph["edges"], dtype=torch.long)
        peptide_edge_attrs = torch.tensor(
            peptide_graph["edge_attrs"], dtype=torch.long
        )
        peptide_labels = torch.tensor(
            self.peptide_labels[pep_id], dtype=torch.long
        )

        protein_embedding = torch.tensor(
            np.load(Path(f["protein_embedding_dir"]) / f"{prot_id}.npy"),
            dtype=torch.float32,
        )
        protein_edges = torch.tensor(protein_graph["edge"], dtype=torch.long)
        protein_edge_attrs = torch.tensor(
            protein_graph["edge_attr"], dtype=torch.long
        )
        protein_labels = torch.tensor(
            self.protein_labels[prot_key], dtype=torch.long
        )
        protein_mask = torch.tensor(
            np.load(Path(f["protein_mask_dir"]) / f"{prot_id}.npy"),
            dtype=torch.long,
        )

        return (
            peptide_edge_attrs,
            peptide_edges,
            peptide_embedding,
            peptide_index,
            peptide_labels,
            protein_edge_attrs,
            protein_edges,
            protein_labels,
            protein_embedding,
            protein_mask,
        )

