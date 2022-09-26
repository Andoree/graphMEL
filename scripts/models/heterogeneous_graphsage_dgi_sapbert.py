import logging

import torch
import torch.nn as nn
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast

from graphmel.scripts.models.heterogeneous_graphsage_sapbert import HeterogeneousGraphSAGE
from graphmel.scripts.self_alignment_pretraining.dgi import Float32DeepGraphInfomaxV2


class HeteroGraphSageDgiSapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, num_graphsage_layers: int, graphsage_hidden_channels: int,
                 graphsage_dropout_p: float, dgi_loss_weight: float, use_cuda, loss,
                 multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

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

        self.hetero_graphsage = HeterogeneousGraphSAGE(num_layers=num_graphsage_layers,
                                                       hidden_channels=graphsage_hidden_channels,
                                                       dropout_p=graphsage_dropout_p)


        self.dgi = Float32DeepGraphInfomaxV2(
            hidden_channels=graphsage_hidden_channels, encoder=self.hetero_graphsage,
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

    def initialize_graphsage_layers(self, x_dict, edge_index_dict):
        self.eval()
        with torch.no_grad():
            for conv in self.hetero_graphsage.convs:
                conv(x_dict, edge_index_dict)

    def summary_fn(self, x, ):
        return torch.sigmoid(torch.mean(x, dim=0))

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
    def dgi_loss(self, x_dict, edge_index_dict, batch_size, ):

        cor_x_dict = self.corruption_fn(x_dict=x_dict, )

        pos_embs = self.hetero_graphsage(x_dict=x_dict, edge_index_dict=edge_index_dict, )["SRC"]
        assert pos_embs.size(0) == batch_size
        neg_embs = self.hetero_graphsage(x_dict=cor_x_dict, edge_index_dict=edge_index_dict, )["SRC"]
        assert neg_embs.size(0) == batch_size
        summary = self.summary_fn(pos_embs, )

        dgi_loss = self.dgi.loss(pos_embs, neg_embs, summary)

        return dgi_loss

    @autocast()
    def forward(self, query_embed1, query_embed2, concept_ids, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        query_embed1 = query_embed1[:batch_size]
        query_embed2 = query_embed2[:batch_size]

        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        return sapbert_loss

    @autocast()
    def sapbert_loss(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
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
