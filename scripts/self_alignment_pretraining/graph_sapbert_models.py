import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast
from torch_geometric.nn import SAGEConv, FastRGCNConv, RGCNConv, GCNConv, GATv2Conv

from graphmel.scripts.self_alignment_pretraining.dgi import Float32DeepGraphInfomax


class GraphSAGESapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, use_cuda, loss, num_graphsage_layers, num_graphsage_channels,
                 num_inner_graphsage_layers, graphsage_dropout_p, graph_loss_weight, multigpu_flag,
                 use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(GraphSAGESapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.num_graphsage_layers = num_graphsage_layers
        self.num_inner_graphsage_layers = num_inner_graphsage_layers
        self.convs = nn.ModuleList()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.graph_loss_weight = graph_loss_weight
        # if modalities_aggr_type not in ("base", "mean"):
        #     raise ValueError(f"Invalid modalities aggregation type: {modalities_aggr_type}")
        # self.modalities_aggr_type = modalities_aggr_type

        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_dropout_p = graphsage_dropout_p
        for i in range(num_graphsage_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_graphsage_layers):
                src_dim = self.bert_hidden_dim
                trg_dim = self.bert_hidden_dim if (j == 0 and i == 0) else num_graphsage_channels
                # in_channels = self.bert_hidden_dim if (j == 0 and i == 0) or \
                #                                       (i == num_graphsage_layers - 1
                #                                        and j == num_inner_graphsage_layers - 1) \
                #     else num_graphsage_channels
                # in_channels = self.bert_hidden_dim if (j == 0 and i == 0) else num_graphsage_channels
                in_channels = (src_dim, trg_dim)
                sage_conv = SAGEConv(in_channels, num_graphsage_channels)
                inner_convs.append(sage_conv)

            self.convs.append(inner_convs)

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    def encode_tokens(self, embs, adjs):

        for i, ((edge_index, _, size), inner_convs_list) in enumerate(zip(adjs, self.convs)):
            if i == 0:
                x_target = embs[:size[1]]
            for j, conv in enumerate(inner_convs_list):
                # x_target = x[:size[1]]  # Target nodes are always placed first.
                x_target = conv((embs, x_target), edge_index)
                if not (i == self.num_graphsage_layers - 1 and j == self.num_inner_graphsage_layers - 1):
                    x_target = F.dropout(x_target, p=self.graphsage_dropout_p, training=self.training)
                x_target = x_target.relu()
        return x_target

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids, adjs,
                batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        text_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                        return_dict=True)['last_hidden_state'][:, 0]
        text_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                        return_dict=True)['last_hidden_state'][:, 0]

        graph_embed1 = self.encode_tokens(embs=text_embed1, adjs=adjs)[:batch_size]
        graph_embed2 = self.encode_tokens(embs=text_embed2, adjs=adjs)[:batch_size]

        text_embed = torch.cat([text_embed1[:batch_size], text_embed2[:batch_size]], dim=0)
        graph_embed = torch.cat([graph_embed1, graph_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        if self.use_miner:
            hard_pairs_text = self.miner(text_embed, labels)
            hard_pairs_graph = self.miner(graph_embed, labels)
            text_loss = self.loss(text_embed, labels, hard_pairs_text)
            graph_loss = self.loss(graph_embed, labels, hard_pairs_graph)
            loss = text_loss + self.graph_loss_weight * graph_loss
            return loss
        else:
            text_loss = self.loss(text_embed, labels, )
            graph_loss = self.loss(graph_embed, labels, )
            loss = text_loss + self.graph_loss_weight * graph_loss
            return loss

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk


class RGCNSapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, num_hidden_channels: int, num_layers: int, num_inner_layers: int,
                 rgcn_dropout_p: float, graph_loss_weight: float,
                 num_relations: int, num_bases: int, num_blocks: int, use_fast_conv: bool, use_cuda, loss,
                 multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(RGCNSapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.num_layers = num_layers
        self.num_inner_layers = num_inner_layers
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        RGCNConvClass = FastRGCNConv if use_fast_conv else RGCNConv
        self.rgcn_dropout_p = rgcn_dropout_p
        self.graph_loss_weight = graph_loss_weight
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                src_dim = self.bert_hidden_dim
                trg_dim = self.bert_hidden_dim if (j == 0 and i == 0) else num_hidden_channels
                in_channels = (src_dim, trg_dim)
                rgcn_conv = RGCNConvClass(in_channels=in_channels, out_channels=num_hidden_channels,
                                          num_relations=num_relations, num_bases=num_bases, num_blocks=num_blocks, )
                inner_convs.append(rgcn_conv)
            self.convs.append(inner_convs)

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    def encode_tokens(self, embs, adjs, rel_types):

        for i, ((edge_index, _, size), inner_convs_list, rel_type) in enumerate(zip(adjs, self.convs, rel_types)):
            if i == 0:
                x_target = embs[:size[1]]  # Target nodes are always placed first.
            for j, conv in enumerate(inner_convs_list):
                x_target = conv((embs, x_target), edge_index=edge_index, edge_type=rel_type)
                if not (i == self.num_layers - 1 and j == self.num_inner_layers):
                    x_target = F.dropout(x_target, p=self.rgcn_dropout_p, training=self.training)
                x_target = x_target.relu()
        return x_target

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                adjs, rel_types, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        text_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                        return_dict=True)['last_hidden_state'][:, 0]
        text_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                        return_dict=True)['last_hidden_state'][:, 0]

        graph_embed1 = self.encode_tokens(embs=text_embed1, adjs=adjs, rel_types=rel_types)[:batch_size]
        graph_embed2 = self.encode_tokens(embs=text_embed2, adjs=adjs, rel_types=rel_types)[:batch_size]

        text_embed = torch.cat([text_embed1[:batch_size], text_embed2[:batch_size]], dim=0)
        graph_embed = torch.cat([graph_embed1, graph_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        if self.use_miner:
            hard_pairs_text = self.miner(text_embed, labels)
            hard_pairs_graph = self.miner(graph_embed, labels)
            text_loss = self.loss(text_embed, labels, hard_pairs_text)
            graph_loss = self.loss(graph_embed, labels, hard_pairs_graph)
            loss = text_loss + self.graph_loss_weight * graph_loss
            return loss
        else:
            text_loss = self.loss(text_embed, labels, )
            graph_loss = self.loss(graph_embed, labels, )
            loss = text_loss + self.graph_loss_weight * graph_loss
            return loss

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk


class GCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, add_self_loops: bool):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=add_self_loops)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index=edge_index)
        x = self.prelu(x)
        return x


class GCNDGISapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, use_cuda, loss, num_gcn_channels: int, add_self_loops: bool,
                 dgi_loss_weight: float, multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all",
                 agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(GCNDGISapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size

        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder

        self.gcn_conv = GCNLayer(self.bert_hidden_dim, num_gcn_channels, add_self_loops=add_self_loops)
        self.dgi = Float32DeepGraphInfomax(
            hidden_channels=num_gcn_channels, encoder=self.gcn_conv,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=self.corruption_fn)
        self.dgi_loss_weight = dgi_loss_weight

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    def encode_tokens(self, input_ids, attention_mask, edge_index):
        x = self.bert_encoder(input_ids, attention_mask=attention_mask,
                              return_dict=True)['last_hidden_state'][:, 0]
        pos_embs, neg_embs, summary = self.dgi(x, edge_index)

        return pos_embs, neg_embs, summary

    def corruption_fn(self, x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids, edge_index,
                batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        q1_pos_embs, q1_neg_embs, q1_summary = self.encode_tokens(input_ids=term_1_input_ids,
                                                                  attention_mask=term_1_att_masks,
                                                                  edge_index=edge_index)
        q2_pos_embs, q2_neg_embs, q2_summary = self.encode_tokens(input_ids=term_2_input_ids,
                                                                  attention_mask=term_2_att_masks,
                                                                  edge_index=edge_index)
        q1_pos_embs, q2_pos_embs = q1_pos_embs[:batch_size], q2_pos_embs[:batch_size]
        q1_neg_embs, q2_neg_embs = q1_neg_embs[:batch_size], q2_neg_embs[:batch_size]

        assert q1_pos_embs.size()[0] == q2_pos_embs.size()[0] == batch_size
        assert q1_neg_embs.size()[0] == q2_neg_embs.size()[0] == batch_size
        query_embed = torch.cat([q1_pos_embs, q2_pos_embs], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        q1_dgi_loss = self.dgi.loss(q1_pos_embs, q1_neg_embs, q1_summary)
        q2_dgi_loss = self.dgi.loss(q2_pos_embs, q2_neg_embs, q2_summary)

        return sapbert_loss + (q1_dgi_loss + q2_dgi_loss) * self.dgi_loss_weight

    @autocast()
    def eval_step_loss(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                       batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)
        return sapbert_loss

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk


class RGCNEncoder(nn.Module):
    def __init__(self, in_channels, num_layers: int, num_hidden_channels, dropout_p: float,
                 use_fast_conv: bool, num_bases: int, num_blocks: int, num_relations: int):
        super().__init__()
        RGCNConvClass = FastRGCNConv if use_fast_conv else RGCNConv
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else num_hidden_channels

            rgcn_conv = RGCNConvClass(in_channels=in_channels, out_channels=num_hidden_channels,
                                      num_relations=num_relations, num_bases=num_bases,
                                      num_blocks=num_blocks, )
            self.convs.append(rgcn_conv)

        self.prelu = nn.PReLU(num_hidden_channels)

    def forward(self, x, edge_index, edge_type, batch_size):
        (x_src, _) = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index=edge_index, edge_type=edge_type)
            x = self.prelu(x)
            if i != self.num_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x_target = x[:batch_size]
                x = (x_src, x_target)

        return x


class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, num_layers: int, num_hidden_channels, dropout_p: float,
                 num_att_heads: int, attention_dropout_p: float, edge_dim: int, ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else num_hidden_channels
            gat_conv = GATv2Conv(in_channels=in_channels, out_channels=num_hidden_channels,
                                 heads=num_att_heads, dropout=attention_dropout_p,
                                 add_self_loops=False, edge_dim=edge_dim, share_weights=True)
            self.convs.append(gat_conv)

        self.prelu = nn.PReLU(num_hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch_size):
        (x_src, _) = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.prelu(x)
            if i != self.num_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x_target = x[:batch_size]
                x = (x_src, x_target)

        return x


class RGCNDGISapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, num_rgcn_layers: int, num_rgcn_channels: int, rgcn_dropout_p: float,
                 dgi_loss_weight: float, num_relations: int, num_bases: int, num_blocks: int, use_fast_conv: bool,
                 use_cuda, loss, multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all",
                 agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(RGCNDGISapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.rgcn_conv = RGCNEncoder(in_channels=self.bert_hidden_dim, num_layers=num_rgcn_layers,
                                     dropout_p=rgcn_dropout_p, num_hidden_channels=num_rgcn_channels,
                                     use_fast_conv=use_fast_conv, num_bases=num_bases, num_blocks=num_blocks,
                                     num_relations=num_relations)
        self.dgi = Float32DeepGraphInfomax(
            hidden_channels=num_rgcn_channels, encoder=self.rgcn_conv,
            summary=self.summary_fn, corruption=self.corruption_fn)
        self.dgi_loss_weight = dgi_loss_weight

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    def summary_fn(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def corruption_fn(self, x, edge_index, edge_type, batch_size):
        (x_source, x_target) = x
        num_target_nodes = x_target.size(0)
        overall_num_nodes = x_source.size(0)
        num_non_target_nodes = overall_num_nodes - num_target_nodes
        x_target_perm = torch.randperm(num_target_nodes)
        x_non_target_perm = torch.randperm(num_non_target_nodes) + num_target_nodes
        x_overall_perm = torch.cat((x_target_perm, x_non_target_perm), dim=0)
        assert x_overall_perm.dim() == 1 and x_overall_perm.size(0) == overall_num_nodes
        x = (x_source[x_overall_perm], x_target[x_target_perm])
        return x, edge_index, edge_type, batch_size

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                edge_index, edge_type, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        q1_target_nodes = query_embed1[:batch_size]
        q1_node_emb_tuple = (query_embed1, q1_target_nodes)
        q2_target_nodes = query_embed2[:batch_size]
        q2_node_emb_tuple = (query_embed2, q2_target_nodes)
        q1_pos_embs, q1_neg_embs, q1_summary = self.dgi(q1_node_emb_tuple, edge_index=edge_index, edge_type=edge_type,
                                                        batch_size=batch_size)
        q2_pos_embs, q2_neg_embs, q2_summary = self.dgi(q2_node_emb_tuple, edge_index=edge_index, edge_type=edge_type,
                                                        batch_size=batch_size)

        assert q1_pos_embs.size()[0] == q2_pos_embs.size()[0] == batch_size
        assert q1_neg_embs.size()[0] == q2_neg_embs.size()[0] == batch_size

        query_embed = torch.cat([q1_target_nodes, q2_target_nodes], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        q1_dgi_loss = self.dgi.loss(q1_pos_embs, q1_neg_embs, q1_summary)
        q2_dgi_loss = self.dgi.loss(q2_pos_embs, q2_neg_embs, q2_summary)

        return sapbert_loss + (q1_dgi_loss + q2_dgi_loss) * self.dgi_loss_weight

    @autocast()
    def eval_step_loss(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                       batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)
        return sapbert_loss

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk


class GATv2DGISapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, gat_num_layers: int, gat_dropout_p: float, gat_num_hidden_channels: int,
                 gat_num_att_heads: int,
                 gat_attention_dropout_p: float, gat_edge_dim: Union[int, None],
                 gat_use_relation_features, num_relations: Union[int, None],
                 dgi_loss_weight: float, use_cuda, loss, multigpu_flag, use_miner=True, miner_margin=0.2,
                 type_of_triplets="all", agg_mode="cls"):

        logging.info(f"Sap_Metric_Learning! use_cuda={use_cuda} loss={loss} use_miner={miner_margin}"
                     f"miner_margin={miner_margin} type_of_triplets={type_of_triplets} agg_mode={agg_mode}")
        logging.info(f"model parameters: hidden_channels={gat_num_hidden_channels}, att_heads={gat_num_att_heads}, "
                     f"att_dropout={gat_attention_dropout_p}, edge_dim={gat_edge_dim}, "
                     f"use_relations={gat_use_relation_features}")
        super(GATv2DGISapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.gat_use_relation_features = gat_use_relation_features

        if self.gat_use_relation_features:
            self.rel_emb = torch.nn.Embedding(num_embeddings=num_relations, embedding_dim=gat_edge_dim, )
        else:
            assert num_relations is None and gat_edge_dim is None

        self.gat_encoder = GATv2Encoder(in_channels=self.bert_hidden_dim, num_layers=gat_num_layers,
                                        num_hidden_channels=gat_num_hidden_channels, dropout_p=gat_dropout_p,
                                        num_att_heads=gat_num_att_heads, attention_dropout_p=gat_attention_dropout_p,
                                        edge_dim=gat_edge_dim)
        self.dgi = Float32DeepGraphInfomax(
            hidden_channels=gat_num_att_heads * gat_num_hidden_channels, encoder=self.gat_encoder,
            summary=self.summary_fn, corruption=self.corruption_fn)
        self.dgi_loss_weight = dgi_loss_weight

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    def summary_fn(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def corruption_fn(self, x, edge_index, edge_attr, batch_size):
        (x_source, x_target) = x
        x = (x_source[torch.randperm(x_source.size(0))], x_target[torch.randperm(x_target.size(0))])
        return x, edge_index, edge_attr, batch_size

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                edge_index, edge_type, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        if self.gat_use_relation_features:
            edge_attr = self.rel_emb(edge_type)
        else:
            edge_attr = None

        q1_target_nodes = query_embed1[:batch_size]
        q1_node_emb_tuple = (query_embed1, q1_target_nodes)
        q2_target_nodes = query_embed2[:batch_size]
        q2_node_emb_tuple = (query_embed2, q2_target_nodes)
        q1_pos_embs, q1_neg_embs, q1_summary = self.dgi(q1_node_emb_tuple, edge_index=edge_index, edge_attr=edge_attr,
                                                        batch_size=batch_size)
        q2_pos_embs, q2_neg_embs, q2_summary = self.dgi(q2_node_emb_tuple, edge_index=edge_index, edge_attr=edge_attr,
                                                        batch_size=batch_size)

        assert q1_pos_embs.size()[0] == q2_pos_embs.size()[0] == batch_size
        assert q1_neg_embs.size()[0] == q2_neg_embs.size()[0] == batch_size

        query_embed = torch.cat([q1_target_nodes, q2_target_nodes], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        q1_dgi_loss = self.dgi.loss(q1_pos_embs, q1_neg_embs, q1_summary)
        q2_dgi_loss = self.dgi.loss(q2_pos_embs, q2_neg_embs, q2_summary)

        return sapbert_loss + (q1_dgi_loss + q2_dgi_loss) * self.dgi_loss_weight

    @autocast()
    def eval_step_loss(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                       batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:batch_size, 0]
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)
        return sapbert_loss

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk


class RGCNDGISapMetricLearningV2(RGCNDGISapMetricLearning):
    def __init__(self, *args, **kwargs):
        super(RGCNDGISapMetricLearningV2, self).__init__(*args, **kwargs)

    def corruption_fn(self, x, edge_index, edge_type):
        (x_source, x_target) = x
        x = (x_source, x_target[torch.randperm(x_target.size(0))])
        return x, edge_index, edge_type

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                edge_index, edge_type, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        q1_target_node_embs = query_embed1[:batch_size]
        q2_target_node_embs = query_embed2[:batch_size]
        query_embed_mean = torch.mean(torch.stack((query_embed1, query_embed2)), dim=0)
        target_node_embs = torch.mean(torch.stack((q1_target_node_embs, q2_target_node_embs)), dim=0)

        node_emb_tuple = (query_embed_mean, target_node_embs)

        pos_embs, neg_embs, summary = self.dgi(node_emb_tuple, edge_index=edge_index, edge_type=edge_type,
                                               batch_size=batch_size)

        assert pos_embs.size()[0] == neg_embs.size()[0] == batch_size

        query_embed = torch.cat([q1_target_node_embs, q2_target_node_embs], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        dgi_loss = self.dgi.loss(pos_embs, neg_embs, summary)

        return sapbert_loss + dgi_loss * self.dgi_loss_weight


class GATv2DGISapMetricLearningV2(GATv2DGISapMetricLearning):
    def __init__(self, *args, **kwargs):
        super(GATv2DGISapMetricLearningV2, self).__init__(*args, **kwargs)

    def summary_fn(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def corruption_fn(self, x, edge_index, edge_attr, batch_size):
        (x_source, x_target) = x
        x = (x_source, x_target[torch.randperm(x_target.size(0))])
        return x, edge_index, edge_attr, batch_size

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                edge_index, edge_type, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        if self.gat_use_relation_features:
            edge_attr = self.rel_emb(edge_type)
        else:
            edge_attr = None

        q1_target_node_embs = query_embed1[:batch_size]
        q2_target_node_embs = query_embed2[:batch_size]
        query_embed_mean = torch.mean(torch.stack((query_embed1, query_embed2)), dim=0)
        target_node_embs = torch.mean(torch.stack((q1_target_node_embs, q2_target_node_embs)), dim=0)

        node_emb_tuple = (query_embed_mean, target_node_embs)

        pos_embs, neg_embs, summary = self.dgi(node_emb_tuple, edge_index=edge_index, edge_attr=edge_attr,
                                               batch_size=batch_size)

        assert pos_embs.size()[0] == pos_embs.size()[0] == batch_size

        query_embed = torch.cat([q1_target_node_embs, q2_target_node_embs], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        dgi_loss = self.dgi.loss(pos_embs, neg_embs, summary)

        return sapbert_loss + dgi_loss * self.dgi_loss_weight
