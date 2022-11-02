import logging
from typing import Dict

import torch
import torch.nn as nn
from pytorch_metric_learning import miners, losses
from torch import nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

from graphmel.scripts.models.abstract_graphsapbert_model import AbstractGraphSapMetricLearningModel
from graphmel.scripts.models.modules import ModalityDistanceLoss
from graphmel.scripts.self_alignment_pretraining.dgi import Float32DeepGraphInfomaxV2



class HeterogeneousGraphSAGE(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, dropout_p, out_channels, set_out_input_dim_equal: bool):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # TODO: Оставил так, но может быть всё-таки подумать: точно должна быть ReLU?
        self.relu = nn.ReLU(inplace=True)

        self.dropout_p = dropout_p
        for i in range(num_layers):
            if set_out_input_dim_equal and (i == num_layers - 1):
                output_num_channels = out_channels
            else:
                output_num_channels = hidden_channels
            self.convs.append(SAGEConv((-1, -1), output_num_channels))

    @autocast()
    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                self.relu(x_dict["SRC"])
                F.dropout(x_dict["SRC"], p=self.dropout_p, training=self.training, inplace=True)
        return x_dict


class HeteroGraphSAGENeighborsSapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel):
    def __init__(self, bert_encoder, num_graphsage_layers: int, graphsage_hidden_channels: int,
                 graph_loss_weight: float, intermodal_loss_weight: float, graphsage_dropout_p: float, use_cuda,
                 loss, multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls",
                 sapbert_loss_weight: float = 1.0, modality_distance=None):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}"
                .format(use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
                        ))
        logging.info(f"HeteroGraphSAGE + SapBERT model parameters: num_graphsage_layers {num_graphsage_layers}, "
                     f"graphsage_hidden_channels: {graphsage_hidden_channels}.")
        super(HeteroGraphSAGENeighborsSapMetricLearning, self).__init__()
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
        self.graph_loss_weight = graph_loss_weight
        self.sapbert_loss_weight = sapbert_loss_weight
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

        self.graphsage_hidden_channels = graphsage_hidden_channels
        self.num_graphsage_layers = num_graphsage_layers
        self.graphsage_dropout_p = graphsage_dropout_p
        self.hetero_graphsage = HeterogeneousGraphSAGE(num_layers=num_graphsage_layers,
                                                       hidden_channels=graphsage_hidden_channels,
                                                       dropout_p=graphsage_dropout_p)

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

    def initialize_graphsage_layers(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            for conv in self.hetero_graphsage.convs:
                conv(x_dict, edge_index_dict)

    @autocast()
    def bert_encode(self, input_ids, att_masks):
        emb = self.bert_encoder(input_ids, attention_mask=att_masks,
                                return_dict=True)['last_hidden_state'][:, 0]

        return emb

    def graph_encode(self, x_dict, edge_index_dict, batch_size):

        graph_emb = self.hetero_graphsage(x_dict=x_dict, edge_index_dict=edge_index_dict, )

        graph_emb = graph_emb["SRC"]
        assert graph_emb.size(0) == batch_size
        return graph_emb

    @autocast()
    def forward(self, text_embed_1, text_embed_2, concept_ids,
                graph_embed_1, graph_embed_2, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        labels = torch.cat([concept_ids, concept_ids], dim=0)
        text_loss = self.calculate_sapbert_loss(text_embed_1, text_embed_2, labels, batch_size)
        graph_loss = self.calculate_sapbert_loss(graph_embed_1, graph_embed_2, labels, batch_size)
        intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, graph_embed_1, graph_embed_2,
                                                         labels, batch_size)

        return text_loss, graph_loss, intermodal_loss
