import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

from torch_geometric.nn import SAGEConv
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances


class GraphSAGESapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, use_cuda,  loss, num_graphsage_layers, num_graphsage_channels,
                 graphsage_dropout_p, multigpu_flag, use_miner=True, miner_margin=0.2, type_of_triplets="all",
                 agg_mode="cls"):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(GraphSAGESapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        # self.pairwise = pairwise
        # self.learning_rate = learning_rate
        # self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.num_graphsage_layers = num_graphsage_layers
        self.convs = nn.ModuleList()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        # TODO: Попробовать всё-таки ещё раз и свёртку угнать в DataParallel?
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_dropout_p = graphsage_dropout_p
        for i in range(num_graphsage_layers):
            in_channels = self.bert_hidden_dim if i == 0 else num_graphsage_channels
            self.convs.append(SAGEConv(in_channels, num_graphsage_channels))

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
        print("encode tokens x", x.size())
        for i, (edge_index, _, size) in enumerate(adjs):
            print("i, x", i, x.size())
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_graphsage_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.graphsage_dropout_p, training=self.training)
        print("x", x.size())
        return x

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids, adjs,
                batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        logging.info("B")
        query_embed1 = self.encode_tokens(input_ids=term_1_input_ids, attention_mask=term_1_att_masks,
                                          adjs=adjs)[:batch_size]
        logging.info("BB")
        query_embed2 = self.encode_tokens(input_ids=term_2_input_ids, attention_mask=term_2_att_masks,
                                          adjs=adjs)[:batch_size]
        logging.info("BBB")
        print("query_embed1", query_embed1.size())
        print("query_embed2", query_embed2.size())
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        print("query_embed", query_embed.size())
        labels = torch.cat([concept_ids, concept_ids], dim=0)
        print("labels", labels.size())
        logging.info("BBBB")
        if self.use_miner:
            logging.info("C")
            hard_pairs = self.miner(query_embed, labels)
            logging.info("CC")
            return self.loss(query_embed, labels, hard_pairs)
        else:
            return self.loss(query_embed, labels)

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        if self.use_cuda:
            targets = targets.cuda()
        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table
