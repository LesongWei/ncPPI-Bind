# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:08:35 2021

@author: Osama
"""


from torch.utils.data import Dataset
from Bio.PDB import Polypeptide
import numpy as np
import torch
import json, os, re, pickle

    



class graphSitePPI(Dataset):
    
    def __init__(self, data_list):

        self.data_list = data_list

        self.bert_embedding_path = '/work/data1/liutianyuan/wls/pepnn/protT5' 


        with open('/work/data1/liutianyuan/wls/PPI_NonAA/data/protid_to_seq.json', 'r') as f:
            self.protid_to_seq = json.load(f)

        with open('/work/data1/liutianyuan/wls/PPI_NonAA/data/pepid_to_seq.json', 'r') as f:
            self.pepid_to_seq = json.load(f)

        self.prot_graph_path = '/work/data1/liutianyuan/wls/binding_site/protein_graph/10'

        with open('/work/data1/liutianyuan/wls/binding_gt/new_protein_sites_5.json', 'r') as file:
            self.prot_sites = json.load(file)

        self.pep_graph_path = '/work/data1/liutianyuan/wls/binding_site/peptide_graph'

        with open('/work/data1/liutianyuan/wls/binding_gt/peptide_sites_5.json', 'r') as file:
            self.pep_sites = json.load(file)



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
        pdb, pep_id, label = sample[0].split('_')
        pdb_pep_id = pdb + '_' + pep_id
        _, prot_id = sample[1].split('_')

        pdb_prot_id = pdb + '_' + pep_id + '_' + prot_id

        prot_graph_id = pdb_prot_id + ".pkl"


        with open(os.path.join(self.prot_graph_path, prot_graph_id), 'rb') as f:
            prot_graph = pickle.load(f)

        with open(os.path.join(self.pep_graph_path, pdb_pep_id), 'rb') as f:
            pep_graph = pickle.load(f)

        pep_edge_attrs = torch.tensor(pep_graph['edge_attrs'], dtype=torch.int64)
        pep_edges = torch.tensor(pep_graph['edges'])
        pep_node_embed = torch.tensor(pep_graph['node_emb'], dtype=torch.float32)
        pep_node_index = torch.tensor(pep_graph['nodes_residue_index'], dtype=torch.int64)
        pep_sites = torch.tensor(self.pep_sites[pdb_pep_id], dtype=torch.int64)


        
        
        prot_edge_attrs = prot_graph['edge_attr'].long()
        prot_edges = prot_graph['edge']
        prot_sites = torch.tensor(self.prot_sites[pdb_prot_id], dtype=torch.int64)
        prot_emb = torch.tensor(np.load(os.path.join(self.bert_embedding_path, sample[1] + '.npy')), dtype=torch.float32)


        
 
        return pep_edge_attrs, pep_edges, pep_node_embed, pep_node_index, pep_sites, prot_edge_attrs, prot_edges, prot_sites, prot_emb
