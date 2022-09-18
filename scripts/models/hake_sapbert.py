import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses


class HakeSapMetricLearning(nn.Module):
    def __init__(self, bert_encoder, hake_gamma, hake_modulus_weight, hake_phase_weight, num_relation,
                 hake_adversarial_temperature, use_cuda, loss, multigpu_flag, use_miner=True,
                 miner_margin=0.2, type_of_triplets="all", agg_mode="cls", ):

        logging.info(f"Sap_Metric_Learning! use_cuda={use_cuda} loss={loss} use_miner={miner_margin}"
                     f"miner_margin={miner_margin} type_of_triplets={type_of_triplets} agg_mode={agg_mode}")
        logging.info(f"model parameters: hake_gamma: {hake_gamma}, hake_modulus_weight: {hake_modulus_weight}, "
                     f"hake_phase_weight: {hake_phase_weight}, num_relation: {num_relation}, "
                     f"hake_adversarial_temperature: {hake_adversarial_temperature}")
        super(HakeSapMetricLearning, self).__init__()
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.num_relation = num_relation

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        self.epsilon = 2.0
        self.hake_adversarial_temperature = hake_adversarial_temperature

        self.hake_gamma = nn.Parameter(
            torch.Tensor([hake_gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.hake_gamma.item() + self.epsilon) / self.bert_hidden_dim]),
            requires_grad=False
        )

        self.relation_embedding = nn.Embedding(self.num_relation, self.bert_hidden_dim * 3)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding.weight[:, self.bert_hidden_dim:2 * self.bert_hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding.weight[:, 2 * self.bert_hidden_dim:3 * self.bert_hidden_dim]
        )

        self.hake_phase_weight = nn.Parameter(torch.Tensor([[hake_phase_weight * self.embedding_range.item()]]))
        self.hake_modulus_weight = nn.Parameter(torch.Tensor([[hake_modulus_weight]]))

        self.pi = 3.14159262358979323846

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

    def calculate_hake_score(self, head_embeddings: torch.Tensor, rel_embeddings: torch.Tensor,
                             tail_embeddings: torch.Tensor, batch_type):

        phase_head, mod_head = head_embeddings, head_embeddings.clone()
        # print('rel', rel_embeddings.size())
        phase_relation, mod_relation, bias_relation = torch.chunk(rel_embeddings, 3, dim=2)
        phase_tail, mod_tail = tail_embeddings, tail_embeddings.clone()

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == "CORRUPTED_HEAD":
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.hake_phase_weight
        r_score = torch.norm(r_score, dim=2) * self.hake_modulus_weight

        return self.hake_gamma.item() - (phase_score + r_score)

    @autocast()
    def forward(self, term_1_input, term_2_input, concept_ids, model_mode, pos_parent_input=None, rel_ids=None,
                sample_weight=None, neg_rel_corr_h_input=None, neg_rel_corr_t_input=None, pos_child_input=None, ):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        term_1_input_ids, term_1_att_masks = term_1_input
        term_2_input_ids, term_2_att_masks = term_2_input

        query_embed1 = self.bert_encoder(term_1_input_ids, attention_mask=term_1_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        query_embed2 = self.bert_encoder(term_2_input_ids, attention_mask=term_2_att_masks,
                                         return_dict=True)['last_hidden_state'][:, 0]
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        labels = torch.cat([concept_ids, concept_ids], dim=0)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            sapbert_loss = self.loss(query_embed, labels, hard_pairs)
        else:
            sapbert_loss = self.loss(query_embed, labels)

        if model_mode == "validation":
            return sapbert_loss

        pos_parent_input_ids, pos_parent_att_mask = pos_parent_input
        neg_rel_corr_h_input_ids, neg_rel_corr_h_att_mask = neg_rel_corr_h_input
        neg_rel_corr_t_input_ids, neg_rel_corr_t_att_mask = neg_rel_corr_t_input

        assert neg_rel_corr_h_input_ids.dim() == 3
        batch_size = neg_rel_corr_h_input_ids.size()[0]
        negative_sample_size = neg_rel_corr_h_input_ids.size()[1]
        negative_seq_length = neg_rel_corr_h_input_ids.size()[2]
        pos_child_input_ids, pos_child_att_mask = pos_child_input



        pos_parent_emb = \
            self.bert_encoder(pos_parent_input_ids.squeeze(1), attention_mask=pos_parent_att_mask.squeeze(1),
                              return_dict=True)['last_hidden_state'][:, 0].unsqueeze(1)

        neg_rel_corr_h_emb = \
            self.bert_encoder(neg_rel_corr_h_input_ids.view((-1, negative_seq_length)),
                              attention_mask=neg_rel_corr_h_att_mask.view((-1, negative_seq_length)),
                              return_dict=True)['last_hidden_state'][:, 0] \
                .view((batch_size, negative_sample_size, -1))
        neg_rel_corr_t_emb = \
            self.bert_encoder(neg_rel_corr_t_input_ids.view((-1, negative_seq_length)),
                              attention_mask=neg_rel_corr_t_att_mask.view((-1, negative_seq_length)),
                              return_dict=True)['last_hidden_state'][:, 0] \
                .view((batch_size, negative_sample_size, -1))

        pos_child_emb = self.bert_encoder(pos_child_input_ids.squeeze(1), attention_mask=pos_child_att_mask.squeeze(1),
                                          return_dict=True)['last_hidden_state'][:, 0].unsqueeze(1)
        rel_embs = self.relation_embedding(rel_ids).unsqueeze(1)

        score_pos = F.logsigmoid(self.calculate_hake_score(head_embeddings=pos_parent_emb,
                                                                  rel_embeddings=rel_embs,
                                                                  tail_embeddings=pos_child_emb,
                                                                  batch_type="SINGLE")).squeeze(dim=1)

        score_neg_corrupted_tail = self.calculate_hake_score(head_embeddings=pos_parent_emb,
                                                                    rel_embeddings=rel_embs,
                                                                    tail_embeddings=neg_rel_corr_t_emb,
                                                                    batch_type="CORRUPTED_TAIL")
        score_neg_corrupted_tail = (F.softmax(score_neg_corrupted_tail *
                                                     self.hake_adversarial_temperature, dim=1).detach()
                                           * F.logsigmoid(-score_neg_corrupted_tail)).sum(dim=1)

        score_neg_corrupted_head = self.calculate_hake_score(head_embeddings=neg_rel_corr_h_emb,
                                                                    rel_embeddings=rel_embs,
                                                                    tail_embeddings=pos_child_emb,
                                                                    batch_type="CORRUPTED_HEAD")
        score_neg_corrupted_head = (F.softmax(score_neg_corrupted_head *
                                                     self.hake_adversarial_temperature, dim=1).detach()
                                           * F.logsigmoid(-score_neg_corrupted_head)).sum(dim=1)


        weights_sum = sample_weight.sum()
        positive_hake_loss = - (sample_weight * score_pos).sum() / weights_sum
        negative_hake_loss = - (sample_weight * (score_neg_corrupted_head +
                                                                score_neg_corrupted_tail) / 2).sum() / weights_sum

        hake_loss = positive_hake_loss + negative_hake_loss

        return sapbert_loss, hake_loss


def get_loss(self, outputs, targets):
    if self.use_cuda:
        targets = targets.cuda()
    loss, in_topk = self.criterion(outputs, targets)
    return loss, in_topk
