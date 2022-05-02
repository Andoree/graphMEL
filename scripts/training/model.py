import random
from typing import List

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, RGCNConv

from graphmel.scripts.training.layers.distmult import DistMult

try:
    import torch_cluster

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
EPS = 1e-15


class GraphSAGEOverBert(nn.Module):
    def __init__(self, bert_encoder, hidden_channels, num_layers, graphsage_dropout, multigpu_flag):
        super(GraphSAGEOverBert, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_dropout = graphsage_dropout
        for i in range(num_layers):
            in_channels = self.bert_hidden_dim if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, input_ids, attention_mask, adjs):
        last_hidden_states = self.bert_encoder(input_ids, attention_mask=attention_mask,
                                               return_dict=True)['last_hidden_state']
        x = torch.stack([elem[0, :] for elem in last_hidden_states])
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.graphsage_dropout, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.graphsage_dropout, training=self.training)
        return x


class BertOverNode2Vec(torch.nn.Module):

    def __init__(self, bert_encoder, seq_max_length, multigpu_flag):
        super().__init__()

        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.seq_max_length = seq_max_length
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder

    def forward(self, batch=None):
        raise NotImplementedError("Forward in BertOverNode2Vec")

    def bert_encode_rw(self, rw_input_ids, rw_attention_mask):
        # rw_input_ids (batch_size, context_size, seq_length)

        batch_size = rw_input_ids.size()[0]
        context_size = rw_input_ids.size()[1]

        start_token_ids, rest_token_ids = rw_input_ids[:, 0], rw_input_ids[:, 1:].contiguous()
        start_att_mask, rest_att_mask = rw_attention_mask[:, 0], rw_attention_mask[:, 1:].contiguous()

        start_last_hidden_states = self.bert_encoder(start_token_ids, attention_mask=start_att_mask,
                                                     return_dict=True)['last_hidden_state']
        rest_last_hidden_states = self.bert_encoder(rest_token_ids.view(-1, self.seq_max_length),
                                                    attention_mask=rest_att_mask.view(-1, self.seq_max_length),
                                                    return_dict=True)['last_hidden_state']

        h_start = torch.stack([elem[0, :] for elem in start_last_hidden_states]).view(batch_size, 1,
                                                                                      self.bert_hidden_dim)
        h_rest = torch.stack([elem[0, :] for elem in rest_last_hidden_states]).view(batch_size, -1,
                                                                                    self.bert_hidden_dim)

        assert h_rest.size(1) == context_size - 1

        return h_start, h_rest

    def loss(self, pos_rw_input_ids, pos_rw_attention_mask, neg_rw_input_ids, neg_rw_attention_mask):
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss
        h_start, h_rest = self.bert_encode_rw(rw_input_ids=pos_rw_input_ids,
                                              rw_attention_mask=pos_rw_attention_mask)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        # Negative loss
        h_start, h_rest = self.bert_encode_rw(rw_input_ids=neg_rw_input_ids,
                                              rw_attention_mask=neg_rw_attention_mask)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')


class RGCNLinkPredictorOverBert(nn.Module):
    def __init__(self, bert_encoder, num_hidden_channels: List[int], num_layers: int, num_relations: int,
                 num_bases: int, num_blocks: int, multigpu_flag):
        super(RGCNLinkPredictorOverBert, self).__init__()
        assert len(num_hidden_channels) == num_layers
        self.num_layers = num_layers
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.rgcn_conv_1 = RGCNConv(in_channels=self.bert_hidden_dim, out_channels=num_hidden_channels[0],
                                    num_relations=num_relations, num_bases=num_bases, num_blocks=num_blocks, )
        self.rgcn_conv_2 = None
        if num_layers == 2:
            self.rgcn_conv_2 = RGCNConv(in_channels=num_hidden_channels[0], out_channels=num_hidden_channels[1],
                                        num_relations=num_relations, num_bases=num_bases, num_blocks=num_blocks, )
        self.distmult = DistMult(num_rel=num_relations, rel_emb_size=num_hidden_channels[len(num_hidden_channels) - 1])

    def apply_convs(self, x, edge_index, rel_type):
        x = self.rgcn_conv_1(x=x, edge_index=edge_index, edge_type=rel_type)
        if self.num_layers == 2:
            x = self.rgcn_conv_2(x=x, edge_index=edge_index, edge_type=rel_type)
        return x

    def encode_nodes(self, input_ids, attention_mask):
        last_hidden_states = self.bert_encoder(input_ids, attention_mask=attention_mask,
                                               return_dict=True)['last_hidden_state']
        cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        return cls_embeddings

    def forward(self, pos_src_input_ids, pos_src_attention_mask, pos_src_adjs, pos_src_rel_ids,
                pos_trg_input_ids, pos_trg_attention_mask,  pos_trg_adjs, pos_trg_rel_ids,
                neg_src_input_ids, neg_src_attention_mask,neg_src_adjs, neg_src_rel_ids,
                neg_trg_input_ids, neg_trg_attention_mask, neg_trg_adjs, neg_trg_rel_ids,
                rel_ids, inv_rel_ids, batch_size):
        print("pos_src_input_ids", pos_src_input_ids.size())
        print("pos_src_attention_mask", pos_src_attention_mask.size())
        print("pos_trg_input_ids", pos_trg_input_ids.size())
        print("pos_trg_attention_mask", pos_trg_attention_mask.size())
        print("neg_src_input_ids", neg_src_input_ids.size())
        print("neg_src_attention_mask", neg_src_attention_mask.size())
        print("neg_trg_input_ids", neg_trg_input_ids.size())
        print("neg_trg_attention_mask", neg_trg_attention_mask.size())
        print("rel_ids", rel_ids.size())
        print("inv_rel_ids", inv_rel_ids.size())

        pos_src_cls_embeddings = self.encode_nodes(input_ids=pos_src_input_ids, attention_mask=pos_src_attention_mask)
        print("pos_src_cls_embeddings", pos_src_cls_embeddings.size())
        pos_trg_cls_embeddings = self.encode_nodes(input_ids=pos_trg_input_ids, attention_mask=pos_trg_attention_mask)
        print("pos_trg_cls_embeddings", pos_trg_cls_embeddings.size())
        neg_src_cls_embeddings = self.encode_nodes(input_ids=neg_src_input_ids, attention_mask=neg_src_attention_mask)
        print("neg_src_cls_embeddings", neg_src_cls_embeddings.size())
        neg_trg_cls_embeddings = self.encode_nodes(input_ids=neg_trg_input_ids, attention_mask=neg_trg_attention_mask)
        print("neg_trg_cls_embeddings", neg_trg_cls_embeddings.size())
        # pos_edge_nodes_embeddings = (pos_src_cls_embeddings, pos_trg_cls_embeddings)
        # neg_edge_nodes_embeddings = (neg_src_cls_embeddings, neg_trg_cls_embeddings)

        pos_x_uninverted = self.apply_convs(x=(pos_src_cls_embeddings, pos_trg_cls_embeddings),
                                            edge_index=pos_src_adjs, rel_type=pos_src_rel_ids)[:batch_size]
        print("pos_x_uninverted", pos_x_uninverted.size())
        pos_x_inv = self.apply_convs(x=(pos_trg_cls_embeddings, pos_src_cls_embeddings),
                                     edge_index=pos_trg_adjs, rel_type=pos_trg_rel_ids)[:batch_size]
        print("pos_x_inv", pos_x_inv.size())
        neg_x_uninverted = self.apply_convs(x=(neg_src_cls_embeddings, neg_trg_cls_embeddings),
                                            edge_index=neg_src_adjs, rel_type=neg_src_rel_ids)[:batch_size]
        print("neg_x_uninverted", neg_x_uninverted.size())
        neg_x_inv = self.apply_convs(x=(neg_trg_cls_embeddings, neg_src_cls_embeddings),
                                     edge_index=neg_trg_adjs, rel_type=neg_trg_rel_ids)[:batch_size]
        print("neg_x_inv", neg_x_inv.size())

        pos_scores, pos_reg = self.distmult(src_node_embs=pos_x_uninverted, trg_node_embs=pos_x_inv, rel_ids=rel_ids)
        print("pos_scores", pos_scores.size())
        neg_scores, neg_reg = self.distmult(src_node_embs=neg_x_uninverted, trg_node_embs=neg_x_inv, rel_ids=rel_ids)
        print("neg_scores", neg_scores.size())
        return pos_scores, pos_reg, neg_scores, neg_reg

