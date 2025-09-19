from torch import nn
from .modules import *
from torch_geometric.nn import GINConv, GPSConv, global_add_pool
import torch
from torch.nn import BatchNorm1d as BatchNorm
from typing import Any, Dict, Optional
from torch.nn import (
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)




class ReciprocalLayer2(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.peptide_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)
        
        
        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)
        
        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model,
                                                                           d_k, d_v, dropout=dropout)
        
        
        
        self.ffn_pep = FFN(d_model, d_inner)
        
        self.ffn_prot = FFN(d_model, d_inner)

    def forward(self, pep_enc, prot_enc):

        
        prot_enc, prot_attention = self.protein_attention_layer(prot_enc, prot_enc, prot_enc)
        
        
        pep_enc, pep_attention = self.peptide_attention_layer(pep_enc, pep_enc, pep_enc)
        
        
        prot_enc, pep_enc, prot_seq_attention, pep_node_attention = self.reciprocal_attention_layer(prot_enc,
                                                                                   pep_enc,
                                                                                   pep_enc,
                                                                                   prot_enc)
        prot_enc = self.ffn_prot(prot_enc)
        
        pep_enc = self.ffn_pep(pep_enc)


        return prot_enc, pep_enc, prot_attention, pep_attention, prot_seq_attention, pep_node_attention


class GPS(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any] = {}):
        super().__init__()

        self.node_embeder = nn.Linear(in_channels, out_channels)

        self.convs = ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        hidden_channels = out_channels
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(out_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GPSConv(out_channels, GINConv(mlp), heads=3,
                           attn_type=attn_type, attn_kwargs=attn_kwargs, norm="batch_norm", dropout=0.1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

    def forward(self, pre_node_emb, edge_index):

        pre_node_emb = pre_node_emb.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_index = edge_index.t().contiguous() 
        node_emb = self.node_embeder(pre_node_emb)

        node_feature = node_emb

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            node_feats = conv(node_feature, edge_index)
            node_feature = node_feats + node_feature
            # node_feature = F.relu(batch_norm(node_feature))
            node_feature = batch_norm(node_feature)

        return node_feature


class ReciprocalLayer3(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.peptide_gps_layer = GPS(d_model, 1, attn_type='multihead')
        
        self.protein_gps_layer = GPS(d_model, 1, attn_type='multihead')
        
        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model,
                                                                           d_k, d_v, dropout=dropout)
        
        
        
        self.ffn_pep = FFN(d_model, d_inner)
        
        self.ffn_prot = FFN(d_model, d_inner)

    def forward(self, pep_enc, prot_enc, pep_edge_index, prot_edge_index, prot_bin = None, pep_bin = None):

        if prot_bin is not None:
            prot_enc, prot_attention = self.protein_attention_layer(prot_enc, prot_enc, prot_enc, prot_bin)
        else:
            # prot_enc, prot_attention = self.protein_attention_layer(prot_enc, prot_enc, prot_enc)

            prot_enc = self.protein_gps_layer(prot_enc, prot_edge_index)
            prot_enc = prot_enc.unsqueeze(0)
        
        if pep_bin is not None:
            pep_enc, pep_attention = self.peptide_attention_layer(pep_enc, pep_enc, pep_enc, pep_bin)
        else:
            # pep_enc, pep_attention = self.peptide_attention_layer(pep_enc, pep_enc, pep_enc)
            pep_enc = self.peptide_gps_layer(pep_enc, pep_edge_index)
            pep_enc = pep_enc.unsqueeze(0)
        
        
        prot_enc, pep_enc, prot_seq_attention, pep_node_attention = self.reciprocal_attention_layer(prot_enc,
                                                                                   pep_enc,
                                                                                   pep_enc,
                                                                                   prot_enc)
        prot_enc = self.ffn_prot(prot_enc)
        
        pep_enc = self.ffn_pep(pep_enc)


        return prot_enc, pep_enc

    

    
