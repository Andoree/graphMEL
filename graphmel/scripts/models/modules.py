import logging

import torch
from torch import nn as nn
from torch.nn import functional as F, Parameter
from torch_geometric.nn import SAGEConv, FastRGCNConv, RGCNConv, GATv2Conv, DataParallel
from torch_geometric.nn.inits import glorot


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, num_outer_layers: int, num_inner_layers: int, num_hidden_channels,
                 dropout_p: float, set_out_input_dim_equal: bool = False, parallel=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_outer_layers = num_outer_layers
        self.num_inner_layers = num_inner_layers
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.set_out_input_dim_equal = set_out_input_dim_equal

        for i in range(num_outer_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                input_num_channels = in_channels if (j == 0 and i == 0) else num_hidden_channels

                output_num_channels = num_hidden_channels
                if set_out_input_dim_equal and (i == num_outer_layers - 1) and (j == num_inner_layers - 1):
                    output_num_channels = in_channels
                # TODO: Если многопоточная версия упадёт, то вот эти строки всему виной
                if parallel:
                    sage_conv = DataParallel(SAGEConv(input_num_channels, output_num_channels))
                else:
                    sage_conv = SAGEConv(input_num_channels, output_num_channels)
                inner_convs.append(sage_conv)

            self.convs.append(inner_convs)

        self.gelu = nn.GELU()

    def forward(self, embs, adjs, *args, **kwargs):
        x = embs
        for i, ((edge_index, _, size), inner_convs_list) in enumerate(zip(adjs, self.convs)):
            for j, conv in enumerate(inner_convs_list):
                x = conv(x, edge_index)
                if not (i == self.num_outer_layers - 1 and j == self.num_inner_layers - 1):
                    x = F.dropout(x, p=self.dropout_p, training=self.training)

                    x = self.gelu(x)

        return x


class ModalityDistanceLoss(nn.Module):
    def __init__(self, dist_name):
        super(ModalityDistanceLoss, self).__init__()
        self.dist_name = dist_name
        if dist_name == "cosine":
            self.distance = nn.CosineSimilarity()
        elif dist_name == "MSE":
            self.distance = nn.MSELoss()
        else:
            raise AttributeError(f"Invalid dist_name: {self.dist_name}")

    def forward(self, text_emb, graph_emb):
        loss = torch.mean(self.distance(text_emb, graph_emb))
        if self.dist_name == "cosine":
            loss = 1. - loss
        return loss


# class GCNLayer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, add_self_loops: bool):
#         super().__init__()
#         self.conv = GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=add_self_loops)
#         self.prelu = nn.PReLU(hidden_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index=edge_index)
#         x = self.prelu(x)
#         return x


class RGCNEncoder(nn.Module):
    def __init__(self, in_channels, num_outer_layers: int, num_inner_layers: int, num_hidden_channels: int,
                 dropout_p: float, use_fast_conv: bool, num_bases: int, num_blocks: int, num_relations: int,
                 set_out_input_dim_equal: bool):
        super().__init__()
        RGCNConvClass = FastRGCNConv if use_fast_conv else RGCNConv
        self.num_outer_layers = num_outer_layers
        self.num_inner_layers = num_inner_layers
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.set_out_input_dim_equal = set_out_input_dim_equal

        self.convs = nn.ModuleList()
        for i in range(num_outer_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                input_num_channels = in_channels if (j == 0 and i == 0) else num_hidden_channels

                output_num_channels = num_hidden_channels
                if set_out_input_dim_equal and (i == num_outer_layers - 1) and (j == num_inner_layers - 1):
                    output_num_channels = in_channels

                rgcn_conv = RGCNConvClass(in_channels=input_num_channels, out_channels=output_num_channels,
                                          num_relations=num_relations, num_bases=num_bases, num_blocks=num_blocks, )
                inner_convs.append(rgcn_conv)

            self.convs.append(inner_convs)

        self.gelu = nn.GELU()

    def forward(self, embs, adjs, rel_types, batch_size, ):
        x = embs
        for i, ((edge_index, _, size), inner_convs_list, rel_type) in enumerate(zip(adjs, self.convs, rel_types)):
            for j, conv in enumerate(inner_convs_list):
                x = conv(x, edge_index=edge_index, edge_type=rel_type)
                if not (i == self.num_outer_layers - 1 and j == self.num_inner_layers - 1):
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
                    x = self.gelu(x)

        return x


class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, num_outer_layers: int, num_inner_layers: int, num_hidden_channels, dropout_p: float,
                 num_relations: int, num_att_heads: int, attention_dropout_p: float, set_out_input_dim_equal,
                 add_self_loops, use_relational_features, remove_activations, layernorm_output):
        super().__init__()
        self.num_outer_layers = num_outer_layers

        self.num_inner_layers = num_inner_layers
        self.num_att_heads = num_att_heads
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.convs = nn.ModuleList()
        for i in range(num_outer_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                input_num_channels = in_channels if (j == 0 and i == 0) else num_hidden_channels

                output_num_channels = num_hidden_channels
                if set_out_input_dim_equal and (i == num_outer_layers - 1) and (j == num_inner_layers - 1):
                    output_num_channels = in_channels
                assert output_num_channels % num_att_heads == 0
                gat_head_output_size = output_num_channels // num_att_heads
                gat_conv = GATv2Conv(in_channels=input_num_channels, out_channels=gat_head_output_size,
                                     heads=num_att_heads, dropout=attention_dropout_p,
                                     add_self_loops=add_self_loops, edge_dim=in_channels, share_weights=True)
                inner_convs.append(gat_conv)

            self.convs.append(inner_convs)
        self.use_relational_features = use_relational_features
        self.remove_activations = remove_activations
        self.gelu = nn.GELU()
        self.lin_proj = None
        if not self.remove_activations:
            self.lin_proj = nn.Linear(in_channels, in_channels)
        if self.use_relational_features:
            self.relation_matrices = Parameter(
                torch.Tensor(num_outer_layers, num_inner_layers, num_relations, in_channels * 2, in_channels))
            glorot(self.relation_matrices)
        self.layernorm_output = layernorm_output
        if self.layernorm_output:
            self.out = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm([in_channels, ], eps=1e-12, elementwise_affine=True),
                nn.Dropout(dropout_p)
            )

    def create_edge_attrs(self, embs, adjs, edge_type_list):
        edge_attrs_list = []

        rel_matrices_list = [self.relation_matrices[edge_type] for edge_type in edge_type_list]
        for (edge_index, _, size), rel_matrices in zip(adjs, rel_matrices_list):
            edge_src_node_ids = edge_index[0]
            edge_trg_node_ids = edge_index[1]

            e_src_emb = embs[edge_src_node_ids]
            e_trg_emb = embs[edge_trg_node_ids]

            assert e_src_emb.dim() == e_trg_emb.dim() == 2

            e_emb = torch.cat((e_src_emb, e_trg_emb), dim=1)
            e_emb = torch.bmm(e_emb.unsqueeze(1), rel_matrices).squeeze(1)
            if self.remove_activations:
                e_emb = e_emb
            else:
                e_emb = self.gelu(e_emb)

            edge_attrs_list.append(e_emb)
        return edge_attrs_list

    def get_edge_feature_matrix(self, embs, relation_matrices, edge_index, edge_type):

        e_emb = torch.cat((embs[edge_index[0]], embs[edge_index[1]]), dim=1)
        rel_matrices = relation_matrices[edge_type]
        e_emb = self.gelu(torch.bmm(e_emb.unsqueeze(1), rel_matrices).squeeze(1))

        return e_emb

    def forward(self, embs, adjs, edge_type_list, batch_size):
        x = embs
        for i, ((edge_index, _, size), inner_convs_list, rel_type) in enumerate(
                zip(adjs, self.convs, edge_type_list)):
            for j, conv in enumerate(inner_convs_list):
                edge_attr = None
                if self.use_relational_features:
                    edge_attr = self.get_edge_feature_matrix(embs=x, relation_matrices=self.relation_matrices[i][j],
                                                             edge_index=edge_index, edge_type=rel_type)
                x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
                if not (i == self.num_outer_layers - 1 and j == self.num_inner_layers - 1):
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
                    if not self.remove_activations:
                        x = self.gelu(x)
        if self.layernorm_output:
            x = self.out(x[:batch_size])
        elif not self.remove_activations:
            x = self.lin_proj(x)
        return x
