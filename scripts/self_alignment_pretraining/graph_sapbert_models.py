import logging

import torch
import torch.nn as nn
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast

from graphmel.scripts.models.abstract_graphsapbert_model import AbstractGraphSapMetricLearningModel
from graphmel.scripts.models.modules import RGCNEncoder, ModalityDistanceLoss, GraphSAGEEncoder


class GraphSAGESapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel):
    def __init__(self, bert_encoder, use_cuda, loss, num_graphsage_layers, num_graphsage_channels,
                 num_inner_graphsage_layers, graphsage_dropout_p, graph_loss_weight, intermodal_loss_weight,
                 multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls",
                 modality_distance=None, sapbert_loss_weight: float = 1.0):

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
        # self.convs = nn.ModuleList()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.sapbert_loss_weight = sapbert_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.intermodal_loss_weight = intermodal_loss_weight
        self.modality_distance = modality_distance
        if modality_distance == "sapbert":
            if self.use_miner:
                self.intermodal_miner = miners.TripletMarginMiner(margin=miner_margin,
                                                                  type_of_triplets=type_of_triplets)
            else:
                self.intermodal_miner = None
            self.intermodal_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

        elif modality_distance is not None:
            self.intermodal_loss = ModalityDistanceLoss(dist_name=modality_distance)
        self.graphhsage_encoder = GraphSAGEEncoder(in_channels=self.bert_hidden_dim,
                                                   num_outer_layers=num_graphsage_layers,
                                                   num_inner_layers=num_inner_graphsage_layers,
                                                   dropout_p=graphsage_dropout_p,
                                                   num_hidden_channels=num_graphsage_channels,
                                                   set_out_input_dim_equal=True)

        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_dropout_p = graphsage_dropout_p

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

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids, adjs,
                batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        text_embed_1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        text_embed_2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        text_loss = self.calculate_sapbert_loss(text_embed_1, text_embed_2, labels, batch_size)

        graph_embed_1 = self.graphhsage_encoder(embs=text_embed_1, adjs=adjs)[:batch_size]
        graph_embed_2 = self.graphhsage_encoder(embs=text_embed_2, adjs=adjs)[:batch_size]

        graph_loss = self.calculate_sapbert_loss(graph_embed_1, graph_embed_2, labels, batch_size)
        intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, graph_embed_1, graph_embed_2,
                                                         labels, batch_size)

        return text_loss, graph_loss, intermodal_loss


class RGCNSapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel):
    def __init__(self, bert_encoder, rgcn_num_hidden_channels: int, rgcn_num_outer_layers: int,
                 rgcn_num_inner_layers: int, rgcn_dropout_p: float, sapbert_loss_weight: float,
                 graph_loss_weight: float, intermodal_loss_weight: float, num_relations: int, num_bases: int,
                 num_blocks: int,
                 use_fast_conv: bool, use_cuda, loss, multigpu_flag, use_miner=True, miner_margin=0.2,
                 type_of_triplets="all", agg_mode="cls", modality_distance=None):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(RGCNSapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.num_layers = rgcn_num_outer_layers
        self.num_inner_layers = rgcn_num_inner_layers
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

        self.sapbert_loss_weight = sapbert_loss_weight
        self.rgcn_dropout_p = rgcn_dropout_p
        self.graph_loss_weight = graph_loss_weight
        self.intermodal_loss_weight = intermodal_loss_weight
        self.modality_distance = modality_distance
        if modality_distance == "sapbert":
            if self.use_miner:
                self.intermodal_miner = miners.TripletMarginMiner(margin=miner_margin,
                                                                  type_of_triplets=type_of_triplets)
            else:
                self.intermodal_miner = None
            self.intermodal_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

        elif modality_distance is not None:
            self.intermodal_loss = ModalityDistanceLoss(dist_name=modality_distance)

        self.rgcn_encoder = RGCNEncoder(in_channels=self.bert_hidden_dim, num_outer_layers=rgcn_num_outer_layers,
                                        num_inner_layers=rgcn_num_inner_layers,
                                        num_hidden_channels=rgcn_num_hidden_channels,
                                        dropout_p=rgcn_dropout_p, use_fast_conv=use_fast_conv, num_bases=num_bases,
                                        num_blocks=num_blocks, num_relations=num_relations,
                                        set_out_input_dim_equal=True)

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

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                adjs, rel_types, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        text_embed_1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        text_embed_2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        text_loss = self.calculate_sapbert_loss(text_embed_1, text_embed_2, labels, batch_size)

        graph_embed_1 = self.rgcn_encoder(embs=text_embed_1, adjs=adjs, rel_types=rel_types,
                                          batch_size=batch_size)[:batch_size]
        graph_embed_2 = self.rgcn_encoder(embs=text_embed_2, adjs=adjs, rel_types=rel_types,
                                          batch_size=batch_size)[:batch_size]

        graph_loss = self.calculate_sapbert_loss(graph_embed_1, graph_embed_2, labels, batch_size)
        intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, graph_embed_1, graph_embed_2,
                                                         labels, batch_size)

        return text_loss, graph_loss, intermodal_loss

# class RGCNDGISapMetricLearningV2(RGCNDGISapMetricLearning):
#     def __init__(self, *args, **kwargs):
#         super(RGCNDGISapMetricLearningV2, self).__init__(*args, **kwargs)
#
#     def corruption_fn(self, x, edge_index, rel_types, batch_size):
#         (x_source, x_target) = x
#         x = (x_source, x_target[torch.randperm(x_target.size(0))])
#         return x, edge_index, rel_types, batch_size
#
#     @autocast()
#     def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
#                 adjs, rel_types, batch_size):
#         """
#         query : (N, h), candidates : (N, topk, h)
#
#         output : (N, topk)
#         """
#         query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
#                                          return_dict=True)['last_hidden_state'][:, 0]
#         query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
#                                          return_dict=True)['last_hidden_state'][:, 0]
#         q1_target_node_embs = query_embed1[:batch_size]
#         q2_target_node_embs = query_embed2[:batch_size]
#         query_embed_mean = torch.mean(torch.stack((query_embed1, query_embed2)), dim=0)
#         target_node_embs = torch.mean(torch.stack((q1_target_node_embs, q2_target_node_embs)), dim=0)
#
#         node_emb_tuple = (query_embed_mean, target_node_embs)
#
#         pos_embs, neg_embs, summary = self.dgi(node_emb_tuple, edge_index=adjs, edge_type=rel_types,
#                                                batch_size=batch_size)
#
#         assert pos_embs.size()[0] == neg_embs.size()[0] == batch_size
#
#         query_embed = torch.cat([q1_target_node_embs, q2_target_node_embs], dim=0)
#         labels = torch.cat([concept_ids, concept_ids], dim=0)
#
#         if self.use_miner:
#             hard_pairs = self.miner(query_embed, labels)
#             sapbert_loss = self.loss(query_embed, labels, hard_pairs)
#         else:
#             sapbert_loss = self.loss(query_embed, labels)
#
#         dgi_loss = self.dgi.loss(pos_embs, neg_embs, summary)
#
#         return sapbert_loss + dgi_loss * self.dgi_loss_weight


# class GATv2DGISapMetricLearningV2(GATv2DGISapMetricLearning):
#     def __init__(self, *args, **kwargs):
#         super(GATv2DGISapMetricLearningV2, self).__init__(*args, **kwargs)
#
#     def summary_fn(self, z, *args, **kwargs):
#         return torch.sigmoid(z.mean(dim=0))
#
#     def corruption_fn(self, x, edge_index, edge_attr, batch_size):
#         (x_source, x_target) = x
#         x = (x_source, x_target[torch.randperm(x_target.size(0))])
#         return x, edge_index, edge_attr, batch_size
#
#     @autocast()
#     def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
#                 edge_index, edge_type, batch_size):
#         """
#         query : (N, h), candidates : (N, topk, h)
#
#         output : (N, topk)
#         """
#         query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
#                                          return_dict=True)['last_hidden_state'][:, 0]
#         query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
#                                          return_dict=True)['last_hidden_state'][:, 0]
#         if self.gat_use_relation_features:
#             edge_attr = self.relation_matrices(edge_type)
#         else:
#             edge_attr = None
#
#         q1_target_node_embs = query_embed1[:batch_size]
#         q2_target_node_embs = query_embed2[:batch_size]
#         query_embed_mean = torch.mean(torch.stack((query_embed1, query_embed2)), dim=0)
#         target_node_embs = torch.mean(torch.stack((q1_target_node_embs, q2_target_node_embs)), dim=0)
#
#         node_emb_tuple = (query_embed_mean, target_node_embs)
#
#         pos_embs, neg_embs, summary = self.dgi(node_emb_tuple, edge_index=edge_index, edge_attr=edge_attr,
#                                                batch_size=batch_size)
#
#         assert pos_embs.size()[0] == pos_embs.size()[0] == batch_size
#
#         query_embed = torch.cat([q1_target_node_embs, q2_target_node_embs], dim=0)
#         labels = torch.cat([concept_ids, concept_ids], dim=0)
#
#         if self.use_miner:
#             hard_pairs = self.miner(query_embed, labels)
#             sapbert_loss = self.loss(query_embed, labels, hard_pairs)
#         else:
#             sapbert_loss = self.loss(query_embed, labels)
#
#         dgi_loss = self.dgi.loss(pos_embs, neg_embs, summary)
#
#         return sapbert_loss + dgi_loss * self.dgi_loss_weight
