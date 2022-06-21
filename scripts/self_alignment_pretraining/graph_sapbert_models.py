from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

from torch_geometric.nn import SAGEConv, FastRGCNConv, RGCNConv, DeepGraphInfomax, GCNConv
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances


class GraphSAGESapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, use_cuda, loss, num_graphsage_layers, num_graphsage_channels,
                 graphsage_dropout_p, multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all",
                 agg_mode="cls"):

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
        self.convs = nn.ModuleList()
        self.bert_hidden_dim = bert_encoder.config.hidden_size

        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_dropout_p = graphsage_dropout_p
        for i in range(num_graphsage_layers):
            in_channels = self.bert_hidden_dim if i == 0 else num_graphsage_channels
            # if multigpu_flag:
            #    sage_conv = nn.DataParallel(SAGEConv(in_channels, num_graphsage_channels))
            # else:
            sage_conv = SAGEConv(in_channels, num_graphsage_channels)
            self.convs.append(sage_conv)

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

    def encode_tokens(self, input_ids, attention_mask, adjs):
        # last_hidden_states = self.bert_encoder(**query_toks1, return_dict=True).last_hidden_state
        x = self.bert_encoder(input_ids, attention_mask=attention_mask,
                              return_dict=True)['last_hidden_state'][:, 0]
        # print("encode tokens x", x.size())
        for i, (edge_index, _, size) in enumerate(adjs):
            # print("i, x", i, x.size())
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_graphsage_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.graphsage_dropout_p, training=self.training)
        # print("x", x.size())
        return x

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids, adjs,
                batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        # logging.info("B")
        query_embed1 = self.encode_tokens(input_ids=term_1_input_ids, attention_mask=term_1_att_masks,
                                          adjs=adjs)[:batch_size]
        # logging.info("BB")
        query_embed2 = self.encode_tokens(input_ids=term_2_input_ids, attention_mask=term_2_att_masks,
                                          adjs=adjs)[:batch_size]
        # logging.info("BBB")
        # print("query_embed1", query_embed1.size())
        # print("query_embed2", query_embed2.size())
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        # print("query_embed", query_embed.size())
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        # print("labels", labels.size())
        # logging.info("BBBB")
        if self.use_miner:
            # logging.info("C")
            hard_pairs = self.miner(query_embed, labels)
            # logging.info("CC")
            return self.loss(query_embed, labels, hard_pairs)
        else:
            return self.loss(query_embed, labels)

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk


class RGCNSapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, num_hidden_channels: int, num_layers: int, rgcn_dropout_p: float,
                 num_relations: int, num_bases: int, num_blocks: int, use_fast_conv: bool, use_cuda, loss,
                 multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(RGCNSapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.num_layers = num_layers
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
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.bert_hidden_dim if i == 0 else num_hidden_channels
            rgcn_conv = RGCNConvClass(in_channels=in_channels, out_channels=num_hidden_channels,
                                      num_relations=num_relations, num_bases=num_bases,
                                      num_blocks=num_blocks, )
            self.convs.append(rgcn_conv)

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

    def encode_tokens(self, input_ids, attention_mask, adjs, rel_types):
        x = self.bert_encoder(input_ids, attention_mask=attention_mask,
                              return_dict=True)['last_hidden_state'][:, 0]

        for i, ((edge_index, _, size), rel_type) in enumerate(zip(adjs, rel_types)):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index=edge_index, edge_type=rel_type)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.rgcn_dropout_p, training=self.training)
        return x

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                adjs, rel_types, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        query_embed1 = self.encode_tokens(input_ids=term_1_input_ids, attention_mask=term_1_att_masks, adjs=adjs,
                                          rel_types=rel_types)[:batch_size]
        query_embed2 = self.encode_tokens(input_ids=term_2_input_ids, attention_mask=term_2_att_masks, adjs=adjs,
                                          rel_types=rel_types)[:batch_size]

        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs)
        else:
            return self.loss(query_embed, labels)

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
        self.dgi = DeepGraphInfomax(
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
