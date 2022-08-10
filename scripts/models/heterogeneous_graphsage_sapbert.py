import logging
from typing import Dict

import torch
import torch.nn as nn
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

from graphmel.scripts.self_alignment_pretraining.dgi import Float32DeepGraphInfomaxV2


class HeterogeneousGraphSAGE(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, dropout_p):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout_p = dropout_p
        for i in range(num_layers):
            # num_conv_out_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv((-1, -1), hidden_channels))

    @autocast()
    def forward(self, x_dict, edge_index_dict):
        # logging.info(f"x {type(x)}, {x}")
        # logging.info(f"edge_index {type(edge_index)}, {edge_index}")
        # logging.info(f"x {type(x_dict)}")
        # logging.info(f"edge_index {type(edge_index_dict)}")
        for i, conv in enumerate(self.convs):
            # logging.info(f"i {i} x {type(x_dict)}, edge_index {type(edge_index_dict)} ")
            x_dict = conv(x_dict, edge_index_dict).relu()
            if i != len(self.convs) - 1:
                x_dict = F.dropout(x_dict, p=self.dropout_p, training=self.training)
        # logging.info("Forward is finished")
        return x_dict


class HeteroGraphSAGESapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, num_graphsage_layers: int, graphsage_hidden_channels: int,
                 graphsage_dropout_p: float, dgi_loss_weight: float, use_cuda, loss,
                 multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}"
                .format(use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
                        ))
        logging.info(f"HeteroGraphSAGE + SapBERT model parameters: num_graphsage_layers {num_graphsage_layers}, "
                     f"graphsage_hidden_channels: {graphsage_hidden_channels}.")
        super(HeteroGraphSAGESapMetricLearning, self).__init__()
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
            self.hetero_graphsage(x_dict, edge_index_dict)

    def summary_fn(self, x, ):
        return torch.sigmoid(torch.mean(x, dim=0))

    def x_dict_to_tensor(self, x_dict, batch_size, local_id2batch_id: Dict[str, Dict[int, int]], device):

        embs = torch.zeros((batch_size, self.graphsage_hidden_channels), dtype=torch.float32).to(device)
        weights = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)
        for sem_gr, x in x_dict.items():
            for i in range(x.size(0)):
                x_i = x[i]
                # logging.info(f"i {i} x_i {x_i.size()}")
                if local_id2batch_id.get(sem_gr) is not None:
                    batch_node_id = local_id2batch_id[sem_gr][i]
                    if batch_node_id < batch_size:
                        embs[batch_node_id] += x_i
                        weights[batch_node_id] += 1
        embs = embs / weights
        return embs

    @staticmethod
    def corruption_fn(x_dict, ):
        corr_x_dict = {node_type: x[torch.randperm(x.size(0))] for node_type, x in x_dict.items()}

        return corr_x_dict

    @autocast()
    def bert_encode(self, input_ids, att_masks):
        emb = self.bert_encoder(input_ids, attention_mask=att_masks,
                                return_dict=True)['last_hidden_state'][:, 0]
        # logging.info(f"emb {emb.dtype}")
        return emb

    @autocast()
    def dgi_loss(self, x_dict, edge_index_dict, batch_size, local_id2batch_id, device):
        # pos_embs, neg_embs, summary = self.dgi(x_dict, edge_index_dict)
        pos_embs_dict = self.hetero_graphsage(x_dict=x_dict, edge_index_dict=edge_index_dict, )

        cor_x_dict = self.corruption_fn(x_dict=x_dict, )

        neg_embs_dict = self.hetero_graphsage(x_dict=cor_x_dict, edge_index_dict=edge_index_dict, )

        pos_embs = self.x_dict_to_tensor(pos_embs_dict, batch_size=batch_size, local_id2batch_id=local_id2batch_id,
                                         device=device)

        summary = self.summary_fn(pos_embs, )
        neg_embs = self.x_dict_to_tensor(neg_embs_dict, batch_size=batch_size, local_id2batch_id=local_id2batch_id,
                                         device=device)

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
