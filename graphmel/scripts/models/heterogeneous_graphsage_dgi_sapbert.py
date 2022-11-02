import logging

import torch
import torch.nn as nn
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast

from graphmel.scripts.models.heterogeneous_graphsage_sapbert import HeterogeneousGraphSAGE
from graphmel.scripts.self_alignment_pretraining.dgi import Float32DeepGraphInfomaxV2

from graphmel.scripts.models.abstract_graphsapbert_model import AbstractGraphSapMetricLearningModel
from graphmel.scripts.models.modules import ModalityDistanceLoss


class HeteroGraphSageDgiSapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel):
    def __init__(self, bert_encoder, num_graphsage_layers: int, graphsage_hidden_channels: int,
                 graphsage_dropout_p: float, graph_loss_weight: float, intermodal_loss_weight: float,
                 dgi_loss_weight: float, use_cuda, loss, multigpu_flag, use_intermodal_miner=True, use_miner=True,
                 miner_margin=0.2, type_of_triplets="all", agg_mode="cls",  modality_distance=None,
                 sapbert_loss_weight: float = 1.0,):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}"
                .format(use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
                        ))
        logging.info(f"HeteroGraphSAGE + SapBERT model parameters: num_graphsage_layers {num_graphsage_layers}, "
                     f"graphsage_hidden_channels: {graphsage_hidden_channels}.")
        super(HeteroGraphSageDgiSapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.use_intermodal_miner = use_intermodal_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_hidden_channels = graphsage_hidden_channels
        self.num_graphsage_layers = num_graphsage_layers
        self.graphsage_dropout_p = graphsage_dropout_p
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

        self.hetero_graphsage = HeterogeneousGraphSAGE(num_layers=num_graphsage_layers,
                                                       hidden_channels=graphsage_hidden_channels,
                                                       dropout_p=graphsage_dropout_p,
                                                       out_channels=self.bert_hidden_dim, set_out_input_dim_equal=True)

        self.dgi = Float32DeepGraphInfomaxV2(
            hidden_channels=self.bert_hidden_dim, encoder=self.hetero_graphsage,
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

    def initialize_graphsage_layers(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            for conv in self.hetero_graphsage.convs:
                conv(x_dict, edge_index_dict)


    @staticmethod
    def summary_fn(z, *args, **kwargs):
        batch_size = kwargs.get("batch_size")
        if batch_size is not None:
            z = z[:batch_size]
        return torch.sigmoid(z.mean(dim=0))

    # def summary_fn(self, x, ):
    #     return torch.sigmoid(torch.mean(x, dim=0))

    @staticmethod
    def corruption_fn(x_dict, ):
        corr_x_dict = {node_type: x for node_type, x in x_dict.items() if node_type != "SRC"}
        corr_x_dict["SRC"] = x_dict["SRC"][torch.randperm(x_dict["SRC"].size(0))]

        return corr_x_dict

    @autocast()
    def bert_encode(self, input_ids, att_masks):
        emb = self.bert_encoder(input_ids, attention_mask=att_masks,
                                return_dict=True)['last_hidden_state'][:, 0]

        return emb

    @autocast()
    def graph_encode(self, x_dict, edge_index_dict, batch_size):

        cor_x_dict = self.corruption_fn(x_dict=x_dict, )

        pos_embs = self.hetero_graphsage(x_dict=x_dict, edge_index_dict=edge_index_dict, )["SRC"]
        assert pos_embs.size(0) == batch_size
        neg_embs = self.hetero_graphsage(x_dict=cor_x_dict, edge_index_dict=edge_index_dict, )["SRC"]
        assert neg_embs.size(0) == batch_size
        summary = self.summary_fn(pos_embs, )

        return pos_embs, neg_embs, summary


    # @autocast()
    # def dgi_loss(self, x_dict, edge_index_dict, batch_size, ):
    #
    #     cor_x_dict = self.corruption_fn(x_dict=x_dict, )
    #
    #     pos_embs = self.hetero_graphsage(x_dict=x_dict, edge_index_dict=edge_index_dict, )["SRC"]
    #     assert pos_embs.size(0) == batch_size
    #     neg_embs = self.hetero_graphsage(x_dict=cor_x_dict, edge_index_dict=edge_index_dict, )["SRC"]
    #     assert neg_embs.size(0) == batch_size
    #     summary = self.summary_fn(pos_embs, )
    #
    #     dgi_loss = self.dgi.loss(pos_embs, neg_embs, summary)
    #
    #     return dgi_loss

    @autocast()
    def forward(self, text_embed_1, text_embed_2, concept_ids, pos_graph_embed_1, pos_graph_embed_2,
                neg_graph_embed_1, neg_graph_embed_2, graph_summary_1, graph_summary_2, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        labels = torch.cat([concept_ids, concept_ids], dim=0)
        sapbert_loss = self.calculate_sapbert_loss(text_embed_1, text_embed_2, labels, batch_size)

        graph_loss = self.calculate_sapbert_loss(pos_graph_embed_1, pos_graph_embed_2, labels, batch_size)

        dgi_loss_1 = self.dgi.loss(pos_graph_embed_1, neg_graph_embed_1, graph_summary_1)
        dgi_loss_2 = self.dgi.loss(pos_graph_embed_2, neg_graph_embed_2, graph_summary_2)
        dgi_loss = (dgi_loss_1 + dgi_loss_2) / 2

        intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, pos_graph_embed_1,
                                                         pos_graph_embed_2, labels, batch_size)

        return sapbert_loss, graph_loss, dgi_loss, intermodal_loss

