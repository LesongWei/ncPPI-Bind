import numpy as np
import torch
import torch.nn as nn
from .layers import *
from .modules import *
# from .graphNetwork import GIN, GAT, GIN2
from torch_scatter import scatter_mean, scatter_sum

def to_var(x, device=None):
    if torch.cuda.is_available():
        if device is None:
            # 获取当前设备
            device = torch.cuda.current_device()
        x = x.to(device) 
    return x

class GIN2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        

        self.node_embeder = nn.Linear(in_channels, out_channels)

        # self.emb_layer_norm = BatchNorm1d(out_channels)
        # self.emb_layer_norm = nn.LayerNorm(out_channels)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        hidden_channels = out_channels

        for i in range(num_layers):
            mlp = Sequential(
                Linear(out_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))


    def forward(self, pre_node_emb, edge_index):


        pre_node_emb = pre_node_emb.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_index = edge_index.t().contiguous() 
        node_emb = self.node_embeder(pre_node_emb)

        node_feature = node_emb

        # node_feature = self.emb_layer_norm(node_emb)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            node_feats = conv(node_feature, edge_index)
            node_feature = node_feats + node_feature
            # node_feature = F.relu(batch_norm(node_feature))
            node_feature = batch_norm(node_feature)

        node_feature = node_feature  # [num_nodes, hidden_dim]
        return node_feature

class ScalarMix(nn.Module):
    def __init__(self, n_layers, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_hidden):  # list of [B, L, d]
        norm_w = F.softmax(self.weight, dim=0)          # α_l
        mixed = 0
        for w, h in zip(norm_w, layer_hidden):
            mixed = mixed + w * h
        return self.dropout(mixed)      # [B, L, d]

class RepeatedModule(nn.Module):
    
    def __init__(self, gin_layers, edge_features, node_features, n_layers, d_model,
                 n_head, d_k, d_v, d_inner, dropout=0.1):
        
        super().__init__()
        

            
        self.pep_extractor = GIN2(300, d_model, gin_layers)

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=300)

        self.prot_extractor = GIN2(1024, d_model, gin_layers)

        self.d_model = d_model 
        
        self.reciprocal_layer_stack = nn.ModuleList([
                ReciprocalLayer2(d_model,  d_inner,  n_head, d_k, d_v, dropout=dropout) 
                for _ in range(n_layers)])
    
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pep_scalar = ScalarMix(n_layers)     
        self.prot_scalar = ScalarMix(n_layers)

        self.norm = nn.LayerNorm(300)

        self.max_len_pep  = 50    # 或根据数据上限设置
        self.max_len_prot = 1143
        self.pep_pos_embed  = nn.Embedding(self.max_len_pep,  self.d_model)
        self.prot_pos_embed = nn.Embedding(self.max_len_prot, self.d_model)
        self.pos_drop = nn.Dropout(dropout)

    
    def forward(self, pep_node_emb, pep_edge_index, pep_edge_attr, pep_residue_index, prot_edge_index, prot_edge_attr, prot_node_emb):

        # bert_features [1, len, 1024]
        
        
        sequence_attention_list = []
        
        graph_attention_list = []
        
        graph_seq_attention_list = []
        
        seq_graph_attention_list = []

        pep_layer_embeddings = []
        prot_layer_embeddings = []

        
        pep_atom_feats = self.pep_extractor(pep_node_emb, pep_edge_index)

        pep_residue_index = pep_residue_index.squeeze(0)  # [num_nodes]

        # 执行聚合
        pep_enc = scatter_sum(pep_atom_feats, pep_residue_index, dim=0)
        pep_enc = self.layer_norm(pep_enc)
        

        pep_enc = self.dropout(pep_enc)

        pep_enc = pep_enc.unsqueeze(0)

        prot_enc = self.prot_extractor(prot_node_emb, prot_edge_index)
        
        prot_enc = self.dropout2(prot_enc)
        
        prot_enc = prot_enc.unsqueeze(0)


        Lp = pep_enc.size(1)
        Lr = prot_enc.size(1)
        device = pep_enc.device

        pep_pos = torch.arange(Lp, device=device).clamp_max(self.max_len_pep - 1)
        prot_pos = torch.arange(Lr, device=device).clamp_max(self.max_len_prot - 1)

        pep_enc  = pep_enc  + self.pep_pos_embed(pep_pos).unsqueeze(0)
        prot_enc = prot_enc + self.prot_pos_embed(prot_pos).unsqueeze(0)

        pep_enc  = self.pos_drop(pep_enc)
        prot_enc = self.pos_drop(prot_enc)


        for reciprocal_layer in self.reciprocal_layer_stack:
            
            prot_enc, pep_enc, prot_attention, pep_attention, prot_seq_attention, pep_node_attention =\
                reciprocal_layer(pep_enc, prot_enc)
            
            sequence_attention_list.append(pep_attention)
            
            graph_attention_list.append(prot_attention)
            
            graph_seq_attention_list.append(prot_seq_attention)
            
            seq_graph_attention_list.append(pep_node_attention)

            prot_layer_embeddings.append(prot_enc)
            pep_layer_embeddings.append(pep_enc)

        
         # ---- 2. ScalarMix across layers ----
        #    列表元素形状均为 [1, L, d]，直接送入即可
        # pep_mix  = self.pep_scalar(pep_layer_embeddings)   # [1, Lp, d]
        # prot_mix = self.prot_scalar(prot_layer_embeddings) # [1, Lr, d]

        # （可选）再加残差：pep_mix += pep_layer_embeddings[-1]，看验证效果
        # pep_mix  = self.norm(pep_mix  + pep_layer_embeddings[-1])
        # prot_mix = self.norm(prot_mix + prot_layer_embeddings[-1]) 
        
        
        return prot_enc, pep_enc, sequence_attention_list, graph_attention_list,\
            seq_graph_attention_list, graph_seq_attention_list
    

class FullModel(nn.Module):
    
    def __init__(self, edge_features, node_features, n_layers, d_model, n_head,
                 d_k, d_v, d_inner, dropout=0.1, return_attention=False):
        
        super().__init__()

        self.gin_layers = 3
        self.repeated_module = RepeatedModule(self.gin_layers, edge_features, node_features, n_layers, d_model, 
                                            n_head, d_k, d_v, d_inner, dropout=dropout)
        
      
        self.final_ffn_prot = FFN(d_model, d_inner, dropout=dropout) 

        self.final_ffn_pep = FFN(d_model, d_inner, dropout=dropout)

        
        self.output_projection_port = nn.Linear(d_model, 2)
        self.output_projection_pep = nn.Linear(d_model, 2)

        self.proj_dim = 128
        self.proj_prot = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self.proj_dim)
        )
        self.proj_pep = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self.proj_dim)
        )
        
                
        self.return_attention = return_attention
        
    def forward(self, pep_edge_attrs, pep_edges, pep_node_embed, pep_node_index, prot_edge_attrs, prot_edges, prot_emb):
        
        # print(pep_node_embed.dtype, prot_emb.dtype)
      
        prot_enc, pep_enc, sequence_attention_list, graph_attention_list,\
            seq_graph_attention_list, graph_seq_attention_list = self.repeated_module(pep_node_embed,
                                                                                     pep_edges,
                                                                                     pep_edge_attrs,
                                                                                     pep_node_index,
                                                                                     prot_edges,
                                                                                     prot_edge_attrs,
                                                                                     prot_emb,
                                                                                     )
         
        
        #对prot的节点进行分类
        prot_node_feat = self.final_ffn_prot(prot_enc)

        prot_node_enc = self.output_projection_port(prot_node_feat)

        # 对 pep 的节点进行分类

        pep_node_feat = self.final_ffn_pep(pep_enc)
        
        pep_node_enc = self.output_projection_pep(pep_node_feat)

        prot_node_cls = self.proj_prot(prot_node_feat)
        pep_node_cls = self.proj_pep(pep_node_feat)
        

        return  prot_node_enc, pep_node_enc, prot_node_cls, pep_node_cls

        
