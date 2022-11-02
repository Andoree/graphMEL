import logging

import torch
from pytorch_metric_learning import miners, losses
from torch import nn as nn
from torch.cuda.amp import autocast

from graphmel.scripts.models.abstract_dgi_model import AbstractDGIModel
from graphmel.scripts.models.abstract_graphsapbert_model import AbstractGraphSapMetricLearningModel
from graphmel.scripts.models.modules import ModalityDistanceLoss, RGCNEncoder
from graphmel.scripts.self_alignment_pretraining.dgi import Float32DeepGraphInfomax


class RGCNDGISapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel, AbstractDGIModel):
    def __init__(self, bert_encoder, num_outer_rgcn_layers: int, num_inner_rgcn_layers: int, num_rgcn_channels: int,
                 rgcn_dropout_p: float, graph_loss_weight: float, dgi_loss_weight: float, intermodal_loss_weight: float,
                 num_relations: int, num_bases: int, num_blocks: int, use_fast_conv: bool, use_cuda, loss,
                 multigpu_flag, use_intermodal_miner=True,  use_miner=True, miner_margin=0.2, type_of_triplets="all",
                 agg_mode="cls", modality_distance=None, sapbert_loss_weight: float = 1.0):

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
        self.use_intermodal_miner = use_intermodal_miner
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder

        self.sapbert_loss_weight = sapbert_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.dgi_loss_weight = dgi_loss_weight
        self.intermodal_loss_weight = intermodal_loss_weight
        self.modality_distance = modality_distance
        if modality_distance == "sapbert":
            if self.use_intermodal_miner:
                self.intermodal_miner = miners.TripletMarginMiner(margin=miner_margin,
                                                                  type_of_triplets=type_of_triplets)
            else:
                self.intermodal_miner = None
            self.intermodal_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

        elif modality_distance is not None:
            self.intermodal_loss = ModalityDistanceLoss(dist_name=modality_distance)

        self.rgcn_conv = RGCNEncoder(in_channels=self.bert_hidden_dim, num_outer_layers=num_outer_rgcn_layers,
                                     num_inner_layers=num_inner_rgcn_layers, num_hidden_channels=num_rgcn_channels,
                                     dropout_p=rgcn_dropout_p, num_relations=num_relations, num_blocks=num_blocks,
                                     num_bases=num_bases, use_fast_conv=use_fast_conv,
                                     set_out_input_dim_equal=True)

        self.dgi = Float32DeepGraphInfomax(
            hidden_channels=self.bert_hidden_dim, encoder=self.rgcn_conv,
            summary=self.summary_fn, corruption=self.corruption_fn)

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

        pos_graph_embs_1, neg_graph_embs_1, graph_summary_1, pos_graph_embs_2, neg_graph_embs_2, graph_summary_2 = \
            self.graph_encode(text_embed_1, text_embed_2, adjs=adjs, rel_types=rel_types, batch_size=batch_size)

        dgi_loss_1 = self.dgi.loss(pos_graph_embs_1, neg_graph_embs_1, graph_summary_1)
        dgi_loss_2 = self.dgi.loss(pos_graph_embs_2, neg_graph_embs_2, graph_summary_2)

        graph_loss = self.calculate_sapbert_loss(pos_graph_embs_1, pos_graph_embs_2, labels, batch_size)

        intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, pos_graph_embs_1, pos_graph_embs_2,
                                                         labels, batch_size)

        return text_loss, graph_loss, (dgi_loss_1 + dgi_loss_2) / 2, intermodal_loss