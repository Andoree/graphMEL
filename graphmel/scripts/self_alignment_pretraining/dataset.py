import copy
import random
from typing import List, Dict, Tuple, Any
import logging

from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.loader import NeighborSampler as RawNeighborSampler, NeighborLoader, HGTLoader
from torch_cluster import random_walk
import torch
import numpy as np
from torch_geometric.loader.utils import filter_hetero_data, edge_type_to_str, index_select, filter_node_store_
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor
from tqdm import tqdm

from graphmel.scripts.training.data.data_utils import node_ids2tokenizer_output


class PositivePairNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict,
                 node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(PositivePairNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id_to_token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list

        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):

        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])

        assert len(triplet_concept_ids) == len(term_1_input_ids)
        (batch_size, n_id, adjs) = super(PositivePairNeighborSampler, self).sample(triplet_concept_ids)
        neighbor_node_ids = n_id[batch_size:]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id_to_token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id_to_token_ids_dict,
            seq_max_length=self.seq_max_length)

        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": triplet_concept_ids
        }
        return batch_dict


class PositiveRelationalNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict,
                 rel_ids, node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(PositiveRelationalNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id2token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list
        self.rel_ids = rel_ids
        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

        self.num_edges = self.edge_index.size()[1]

        assert self.num_edges == len(rel_ids)

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])
        assert len(triplet_concept_ids) == len(term_1_input_ids)

        (batch_size, n_id, adjs) = super(PositiveRelationalNeighborSampler, self).sample(triplet_concept_ids)

        neighbor_node_ids = n_id[batch_size:]

        if not isinstance(adjs, list):
            adjs = [adjs, ]
        e_ids_list = [adj.e_id for adj in adjs]
        rel_ids_list = [self.rel_ids[e_ids] for e_ids in e_ids_list]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": triplet_concept_ids, "rel_ids_list": rel_ids_list,
        }
        return batch_dict


class SapMetricLearningHierarchicalDataset(Dataset):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict, node_id2token_ids_dict,
                 concept_id2parents: Dict[int, List[int]], concept_id2childs: Dict[int, List[int]],
                 seq_max_length: int, negative_sample_size: int, ):
        """
        Dataset for training, inherits `torch.utils.data.Dataset`.

        """
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.num_synonym_triplets = len(pos_pairs_term_1_id_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list
        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.node_id2token_ids_dict = node_id2token_ids_dict
        self.seq_max_length = seq_max_length
        self.negative_sample_size = negative_sample_size
        # assert len(concept_id2parents) == len(concept_id2childs)
        self.num_nodes = len(node_id2token_ids_dict)

        # Hierarchy tree fields
        self.concept_id2parents = concept_id2parents
        self.concept_id2children = concept_id2childs

    def __len__(self):
        return self.num_synonym_triplets

    def create_negative_samples(self, concept_id_to_remain: int, concept_id2neighbors: Dict[int, List[int]]) \
            -> np.array:
        neg_triples = []
        neg_size = 0
        neighbors = concept_id2neighbors.get(concept_id_to_remain)
        neighbors = list() if neighbors is None else neighbors

        while neg_size < self.negative_sample_size:
            neg_triples_tmp = np.random.randint(self.num_nodes, size=self.negative_sample_size * 2)

            mask = np.in1d(neg_triples_tmp, neighbors, assume_unique=True, invert=True)

            neg_triples_tmp = neg_triples_tmp[mask]
            neg_triples.append(neg_triples_tmp)
            neg_size += neg_triples_tmp.size
        neg_triples = np.concatenate(neg_triples)[:self.negative_sample_size]

        return neg_triples

    def __getitem__(self, idx: int):
        """
        Returns a positive sample and `self.neg_size` negative samples.
        """

        tokenized_term_1 = self.term_id2tokenizer_output[self.pos_pairs_term_1_id_list[idx]]
        tokenized_term_2 = self.term_id2tokenizer_output[self.pos_pairs_term_2_id_list[idx]]
        term_1_input_ids = tokenized_term_1["input_ids"][0]
        term_1_att_mask = tokenized_term_1["attention_mask"][0]
        term_2_input_ids = tokenized_term_2["input_ids"][0]
        term_2_att_mask = tokenized_term_2["attention_mask"][0]
        anchor_concept_id = self.pos_pairs_concept_ids_list[idx]
        concept_parents_list = self.concept_id2parents.get(anchor_concept_id)
        concept_children_list = self.concept_id2children.get(anchor_concept_id)
        num_parents = len(concept_parents_list) if concept_parents_list is not None else 0
        num_children = len(concept_children_list) if concept_children_list is not None else 0

        rand_float = random.random()
        # TODO: Перепроверить потом, а то зачем-то же эта тудушка тут была
        positive_child_input_ids, positive_child_att_mask = None, None
        positive_parent_input_ids, positive_parent_att_mask = None, None
        anchor_concept_id = torch.LongTensor([anchor_concept_id, ])
        sample_weight = torch.LongTensor([1, ])
        rel_id = torch.LongTensor([0, ])
        if num_children > 0 and num_parents > 0:
            rand_int = random.randint(1, num_children + num_parents)
            if rand_int <= num_children:
                positive_child_concept_id = random.choice(concept_children_list)
                positive_parent_concept_id = anchor_concept_id
                if rand_float < 0.5:
                    positive_parent_input_ids, positive_parent_att_mask = term_1_input_ids, term_1_att_mask
                else:
                    positive_parent_input_ids, positive_parent_att_mask = term_2_input_ids, term_2_att_mask

            else:
                positive_parent_concept_id = random.choice(concept_parents_list)
                positive_child_concept_id = anchor_concept_id
                if rand_float < 0.5:
                    positive_child_input_ids, positive_child_att_mask = term_1_input_ids, term_1_att_mask
                else:
                    positive_child_input_ids, positive_child_att_mask = term_2_input_ids, term_2_att_mask

        elif num_children > 0:
            positive_child_concept_id = random.choice(concept_children_list)
            positive_parent_concept_id = anchor_concept_id
            if rand_float < 0.5:
                positive_parent_input_ids, positive_parent_att_mask = term_1_input_ids, term_1_att_mask
            else:
                positive_parent_input_ids, positive_parent_att_mask = term_2_input_ids, term_2_att_mask

        elif num_parents > 0:
            positive_parent_concept_id = random.choice(concept_parents_list)
            positive_child_concept_id = anchor_concept_id
            if rand_float < 0.5:
                positive_child_input_ids, positive_child_att_mask = term_1_input_ids, term_1_att_mask
            else:
                positive_child_input_ids, positive_child_att_mask = term_2_input_ids, term_2_att_mask
        else:
            positive_parent_concept_id = None
            positive_child_concept_id = None
            sample_weight = torch.LongTensor([0, ])
            rel_id = torch.LongTensor([1, ])
            positive_child_input_ids = torch.zeros(self.seq_max_length, dtype=torch.long)
            positive_child_att_mask = torch.zeros(self.seq_max_length, dtype=torch.long)
            positive_parent_input_ids = torch.zeros(self.seq_max_length, dtype=torch.long)
            positive_parent_att_mask = torch.zeros(self.seq_max_length, dtype=torch.long)

        t1, t2 = positive_child_input_ids is None, positive_child_att_mask is None
        t3, t4 = positive_parent_input_ids is None, positive_parent_att_mask is None
        assert (t1 == t2) and (t3 == t4)
        if t1:
            positive_child_tokenized = random.choice(self.node_id2token_ids_dict[positive_child_concept_id])
            positive_child_input_ids = positive_child_tokenized["input_ids"][0]
            positive_child_att_mask = positive_child_tokenized["attention_mask"][0]
            assert t1 != t3
        if t3:
            positive_parent_tokenized = random.choice(self.node_id2token_ids_dict[positive_parent_concept_id])
            positive_parent_input_ids = positive_parent_tokenized["input_ids"][0]
            positive_parent_att_mask = positive_parent_tokenized["attention_mask"][0]
            assert t1 != t3

        # Positive relation is from parent to child
        negative_relation_corrupted_head = self.create_negative_samples(
            concept_id_to_remain=positive_child_concept_id, concept_id2neighbors=self.concept_id2parents)
        negative_relation_corrupted_tail = self.create_negative_samples(
            concept_id_to_remain=positive_parent_concept_id, concept_id2neighbors=self.concept_id2children)
        negative_relation_corrupted_head_tokenized = \
            [random.choice(self.node_id2token_ids_dict[i]) for i in negative_relation_corrupted_head]
        negative_relation_corrupted_tail_tokenized = \
            [random.choice(self.node_id2token_ids_dict[i]) for i in negative_relation_corrupted_tail]

        negative_relation_corrupted_head_input_ids = torch.stack(
            [d["input_ids"][0] for d in negative_relation_corrupted_head_tokenized])
        negative_relation_corrupted_head_att_mask = torch.stack(
            [d["attention_mask"][0] for d in negative_relation_corrupted_head_tokenized])
        negative_relation_corrupted_tail_input_ids = torch.stack(
            [d["input_ids"][0] for d in negative_relation_corrupted_tail_tokenized])
        negative_relation_corrupted_tail_att_mask = torch.stack(
            [d["attention_mask"][0] for d in negative_relation_corrupted_tail_tokenized])

        sample = {
            "term_1_input_ids": term_1_input_ids, "term_1_att_mask": term_1_att_mask,
            "term_2_input_ids": term_2_input_ids, "term_2_att_mask": term_2_att_mask,
            "anchor_concept_id": anchor_concept_id, "rel_id": rel_id,

            "pos_parent_input_ids": positive_parent_input_ids,
            "pos_parent_att_mask": positive_parent_att_mask,
            "neg_rel_corr_h_input_ids": negative_relation_corrupted_head_input_ids,
            "neg_rel_corr_h_att_mask": negative_relation_corrupted_head_att_mask,
            "neg_rel_corr_t_input_ids": negative_relation_corrupted_tail_input_ids,
            "neg_rel_corr_t_att_mask": negative_relation_corrupted_tail_att_mask,

            "pos_child_input_ids": positive_child_input_ids,
            "pos_child_att_mask": positive_child_att_mask,
            "sample_weight": sample_weight
        }

        return sample

    @staticmethod
    def collate_fn(data):
        term_1_input_ids = torch.stack([d["term_1_input_ids"] for d in data], dim=0)
        term_1_att_mask = torch.stack([d["term_1_att_mask"] for d in data], dim=0)
        term_2_input_ids = torch.stack([d["term_2_input_ids"] for d in data], dim=0)
        term_2_att_mask = torch.stack([d["term_2_att_mask"] for d in data], dim=0)

        anchor_concept_id = torch.stack([d["anchor_concept_id"][0] for d in data], dim=0)
        assert term_1_input_ids.size() == term_1_att_mask.size() == term_2_input_ids.size() == term_2_att_mask.size()

        pos_parent_input_ids = torch.stack([d["pos_parent_input_ids"] for d in data], dim=0).unsqueeze(1)
        pos_parent_att_mask = torch.stack([d["pos_parent_att_mask"] for d in data], dim=0).unsqueeze(1)
        pos_child_input_ids = torch.stack([d["pos_child_input_ids"] for d in data], dim=0).unsqueeze(1)
        pos_child_att_mask = torch.stack([d["pos_child_att_mask"] for d in data], dim=0).unsqueeze(1)
        assert pos_parent_input_ids.size() == pos_parent_att_mask.size() \
               == pos_child_input_ids.size() == pos_child_att_mask.size()

        neg_rel_corr_h_input_ids = torch.stack([d["neg_rel_corr_h_input_ids"] for d in data], dim=0)
        neg_rel_corr_h_att_mask = torch.stack([d["neg_rel_corr_h_att_mask"] for d in data], dim=0)
        neg_rel_corr_t_input_ids = torch.stack([d["neg_rel_corr_t_input_ids"] for d in data], dim=0)
        neg_rel_corr_t_att_mask = torch.stack([d["neg_rel_corr_t_att_mask"] for d in data], dim=0)
        assert neg_rel_corr_h_input_ids.size() == neg_rel_corr_h_att_mask.size() \
               == neg_rel_corr_t_input_ids.size() == neg_rel_corr_t_att_mask.size()

        hierarchical_sample_weight = torch.stack([d["sample_weight"][0] for d in data], dim=0)
        rel_id = torch.stack([d["rel_id"][0] for d in data], dim=0)

        batch = {
            "term_1_input_ids": term_1_input_ids, "term_1_att_mask": term_1_att_mask,
            "term_2_input_ids": term_2_input_ids, "term_2_att_mask": term_2_att_mask,
            "anchor_concept_id": anchor_concept_id, "sample_weight": hierarchical_sample_weight, "rel_id": rel_id,

            "pos_parent_input_ids": pos_parent_input_ids,
            "pos_parent_att_mask": pos_parent_att_mask,
            "pos_child_input_ids": pos_child_input_ids,
            "pos_child_att_mask": pos_child_att_mask,

            "neg_rel_corr_h_input_ids": neg_rel_corr_h_input_ids,
            "neg_rel_corr_h_att_mask": neg_rel_corr_h_att_mask,
            "neg_rel_corr_t_input_ids": neg_rel_corr_t_input_ids,
            "neg_rel_corr_t_att_mask": neg_rel_corr_t_att_mask,

        }

        return batch

    # @staticmethod
    # def collate_fn(data):
    #     term_1_input_ids = torch.stack([d["term_1_input_ids"] for d in data], dim=0)
    #     term_1_att_mask = torch.stack([d["term_1_att_mask"] for d in data], dim=0)
    #     term_2_input_ids = torch.stack([d["term_2_input_ids"] for d in data], dim=0)
    #     term_2_att_mask = torch.stack([d["term_2_att_mask"] for d in data], dim=0)
    #
    #     anchor_concept_id = torch.stack([d["anchor_concept_id"][0] for d in data], dim=0)
    #     assert term_1_input_ids.size() == term_1_att_mask.size() == term_2_input_ids.size() == term_2_att_mask.size()
    #
    #     parent_rel_id = torch.stack([d["parent_rel_id"][0] for d in data], dim=0)
    #     child_rel_id = torch.stack([d["child_rel_id"][0] for d in data], dim=0)
    #     assert term_1_input_ids.size(0) == anchor_concept_id.size(0) == parent_rel_id.size(0) == child_rel_id.size(0)
    #
    #     hierarchical_sample_weight = torch.stack([d["hierarchical_sample_weight"][0] for d in data], dim=0)
    #     # hierarchical_sample_weight = hierarchical_sample_weight > 0
    #     pos_parent_input_ids = torch.stack([d["pos_parent_input_ids"] for d in data], dim=0).unsqueeze(1)
    #     pos_parent_att_mask = torch.stack([d["pos_parent_att_mask"] for d in data], dim=0).unsqueeze(1)
    #
    #     neg_parent_rel_corr_h_input_ids = torch.stack([d["neg_parent_rel_corr_h_input_ids"] for d in data], dim=0)
    #     neg_parent_rel_corr_h_att_mask = torch.stack([d["neg_parent_rel_corr_h_att_mask"] for d in data], dim=0)
    #     neg_parent_rel_corr_t_input_ids = torch.stack([d["neg_parent_rel_corr_t_input_ids"] for d in data], dim=0)
    #     neg_parent_rel_corr_t_att_mask = torch.stack([d["neg_parent_rel_corr_t_att_mask"] for d in data], dim=0)
    #     assert neg_parent_rel_corr_h_input_ids.size() == neg_parent_rel_corr_h_att_mask.size() \
    #            == neg_parent_rel_corr_t_input_ids.size() == neg_parent_rel_corr_t_att_mask.size()
    #
    #     pos_child_input_ids = torch.stack([d["pos_child_input_ids"] for d in data], dim=0).unsqueeze(1)
    #     pos_child_att_mask = torch.stack([d["pos_child_att_mask"] for d in data], dim=0).unsqueeze(1)
    #     assert pos_parent_input_ids.size() == pos_parent_att_mask.size() \
    #            == pos_child_input_ids.size() == pos_child_att_mask.size()
    #
    #     neg_child_rel_corr_h_input_ids = torch.stack([d["neg_child_rel_corr_h_input_ids"] for d in data], dim=0)
    #     neg_child_rel_corr_h_att_mask = torch.stack([d["neg_child_rel_corr_h_att_mask"] for d in data], dim=0)
    #     neg_child_rel_corr_t_input_ids = torch.stack([d["neg_child_rel_corr_t_input_ids"] for d in data], dim=0)
    #     neg_child_rel_corr_t_att_mask = torch.stack([d["neg_child_rel_corr_t_att_mask"] for d in data], dim=0)
    #     assert neg_child_rel_corr_h_input_ids.size() == neg_child_rel_corr_h_att_mask.size() \
    #            == neg_child_rel_corr_t_input_ids.size() == neg_child_rel_corr_t_att_mask.size()
    #
    #     batch = {
    #         "term_1_input_ids": term_1_input_ids, "term_1_att_mask": term_1_att_mask,
    #         "term_2_input_ids": term_2_input_ids, "term_2_att_mask": term_2_att_mask,
    #         "anchor_concept_id": anchor_concept_id,
    #         "parent_rel_id": parent_rel_id, "child_rel_id": child_rel_id,
    #
    #         "hierarchical_sample_weight": hierarchical_sample_weight,
    #         "pos_parent_input_ids": pos_parent_input_ids,
    #         "pos_parent_att_mask": pos_parent_att_mask,
    #         "neg_parent_rel_corr_h_input_ids": neg_parent_rel_corr_h_input_ids,
    #         "neg_parent_rel_corr_h_att_mask": neg_parent_rel_corr_h_att_mask,
    #         "neg_parent_rel_corr_t_input_ids": neg_parent_rel_corr_t_input_ids,
    #         "neg_parent_rel_corr_t_att_mask": neg_parent_rel_corr_t_att_mask,
    #
    #         "pos_child_input_ids": pos_child_input_ids,
    #         "pos_child_att_mask": pos_child_att_mask,
    #         "neg_child_rel_corr_h_input_ids": neg_child_rel_corr_h_input_ids,
    #         "neg_child_rel_corr_h_att_mask": neg_child_rel_corr_h_att_mask,
    #         "neg_child_rel_corr_t_input_ids": neg_child_rel_corr_t_input_ids,
    #         "neg_child_rel_corr_t_att_mask": neg_child_rel_corr_t_att_mask,
    #
    #     }
    #
    #     return batch

    # def __getitem__(self, idx: int):
    #     """
    #     Returns a positive sample and `self.neg_size` negative samples.
    #     """
    #
    #     tokenized_term_1 = self.term_id2tokenizer_output[self.pos_pairs_term_1_id_list[idx]]
    #     tokenized_term_2 = self.term_id2tokenizer_output[self.pos_pairs_term_2_id_list[idx]]
    #     term_1_input_ids = tokenized_term_1["input_ids"][0]
    #     term_1_att_mask = tokenized_term_1["attention_mask"][0]
    #     term_2_input_ids = tokenized_term_2["input_ids"][0]
    #     term_2_att_mask = tokenized_term_2["attention_mask"][0]
    #     anchor_concept_id = self.pos_pairs_concept_ids_list[idx]
    #     concept_parents_list = self.concept_id2parents.get(anchor_concept_id)
    #     # children_of_parent_list = self.concept_id2children.get()
    #     # concept_children_list = self.concept_id2children.get(anchor_concept_id)
    #
    #     # 0 - id of relation from parent to child, 1 - from child to parent
    #     if concept_parents_list is not None and len(concept_parents_list) > 0:
    #         positive_parent_concept_id = random.choice(concept_parents_list)
    #         # children_of_parent_list = self.concept_id2children.get(positive_parent_concept_id)
    #         # pos_relation: (positive_parent_concept_id, IS_A_CHILD_OF, concept_id)
    #         hierarchical_sample_weight = torch.FloatTensor([1, ])
    #
    #         negative_parent_relation_corrupted_heads = \
    #             self.create_negative_samples(concept_id_to_remain=anchor_concept_id,
    #                                          concept_id2neighbors=self.concept_id2parents)
    #         negative_parent_relation_corrupted_tails = \
    #             self.create_negative_samples(concept_id_to_remain=positive_parent_concept_id,
    #                                          concept_id2neighbors=self.concept_id2children)
    #
    #         positive_parent_tokenized = random.choice(self.node_id2token_ids_dict[positive_parent_concept_id])
    #
    #         negative_parent_relation_corrupted_heads_tokenized = \
    #             [random.choice(self.node_id2token_ids_dict[i]) for i in negative_parent_relation_corrupted_heads]
    #         negative_parent_relation_corrupted_tails_tokenized = \
    #             [random.choice(self.node_id2token_ids_dict[i]) for i in negative_parent_relation_corrupted_tails]
    #
    #         positive_parent_input_ids = positive_parent_tokenized["input_ids"][0]
    #         positive_parent_att_mask = positive_parent_tokenized["attention_mask"][0]
    #         negative_parent_relation_corrupted_heads_input_ids = torch.stack(
    #             [d["input_ids"][0] for d in negative_parent_relation_corrupted_heads_tokenized])
    #         negative_parent_relation_corrupted_heads_att_mask = torch.stack(
    #             [d["attention_mask"][0] for d in negative_parent_relation_corrupted_heads_tokenized])
    #
    #         negative_parent_relation_corrupted_tails_input_ids = torch.stack(
    #             [d["input_ids"][0] for d in negative_parent_relation_corrupted_tails_tokenized])
    #         negative_parent_relation_corrupted_tails_att_mask = torch.stack(
    #             [d["attention_mask"][0] for d in negative_parent_relation_corrupted_tails_tokenized])
    #
    #         # Child to parent relation sampling
    #         children_of_parent_list = self.concept_id2children.get(positive_parent_concept_id)
    #         positive_child_of_parent_concept_id = random.choice(children_of_parent_list)
    #
    #         # pos_relation: (positive_child_of_parent_concept_id, IS_A_PARENT_OF, positive_parent_concept_id)
    #         negative_child_relation_corrupted_heads = \
    #             self.create_negative_samples(concept_id_to_remain=positive_parent_concept_id,
    #                                          concept_id2neighbors=self.concept_id2children)
    #         negative_child_relation_corrupted_tails = \
    #             self.create_negative_samples(concept_id_to_remain=positive_child_of_parent_concept_id,
    #                                          concept_id2neighbors=self.concept_id2parents)
    #
    #         positive_child_of_parent_tokenized = \
    #             random.choice(self.node_id2token_ids_dict[positive_child_of_parent_concept_id])
    #         negative_child_relation_corrupted_heads_tokenized = \
    #             [random.choice(self.node_id2token_ids_dict[i]) for i in negative_child_relation_corrupted_heads]
    #         negative_child_relation_corrupted_tails_tokenized = \
    #             [random.choice(self.node_id2token_ids_dict[i]) for i in negative_child_relation_corrupted_tails]
    #
    #         positive_child_of_parent_input_ids = positive_child_of_parent_tokenized["input_ids"][0]
    #         positive_child_of_parent_att_mask = positive_child_of_parent_tokenized["attention_mask"][0]
    #
    #         negative_child_relation_corrupted_heads_input_ids = torch.stack(
    #             [d["input_ids"][0] for d in negative_child_relation_corrupted_heads_tokenized])
    #         negative_child_relation_corrupted_heads_att_mask = torch.stack(
    #             [d["attention_mask"][0] for d in negative_child_relation_corrupted_heads_tokenized])
    #
    #         negative_child_relation_corrupted_tails_input_ids = torch.stack(
    #             [d["input_ids"][0] for d in negative_child_relation_corrupted_tails_tokenized])
    #         negative_child_relation_corrupted_tails_att_mask = torch.stack(
    #             [d["attention_mask"][0] for d in negative_child_relation_corrupted_tails_tokenized])
    #
    #     else:
    #         hierarchical_sample_weight = torch.FloatTensor([0, ])
    #         positive_parent_input_ids = torch.zeros(self.seq_max_length, dtype=torch.long)
    #         positive_parent_att_mask = torch.zeros(self.seq_max_length, dtype=torch.long)
    #         negative_parent_relation_corrupted_heads_input_ids = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #         negative_parent_relation_corrupted_heads_att_mask = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #         negative_parent_relation_corrupted_tails_input_ids = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #         negative_parent_relation_corrupted_tails_att_mask = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #
    #         positive_child_of_parent_input_ids = torch.zeros(self.seq_max_length, dtype=torch.long)
    #         positive_child_of_parent_att_mask = torch.zeros(self.seq_max_length, dtype=torch.long)
    #         negative_child_relation_corrupted_heads_input_ids = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #         negative_child_relation_corrupted_heads_att_mask = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #         negative_child_relation_corrupted_tails_input_ids = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #         negative_child_relation_corrupted_tails_att_mask = \
    #             torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
    #
    #     parent_rel_id = torch.LongTensor([0, ])
    #     child_rel_id = torch.LongTensor([1, ])
    #
    #     anchor_concept_id = torch.LongTensor([anchor_concept_id, ])
    #     sample = {
    #         "term_1_input_ids": term_1_input_ids, "term_1_att_mask": term_1_att_mask,
    #         "term_2_input_ids": term_2_input_ids, "term_2_att_mask": term_2_att_mask,
    #         "anchor_concept_id": anchor_concept_id,
    #         "parent_rel_id": parent_rel_id, "child_rel_id": child_rel_id,
    #
    #         "hierarchical_sample_weight": hierarchical_sample_weight,
    #         "pos_parent_input_ids": positive_parent_input_ids,
    #         "pos_parent_att_mask": positive_parent_att_mask,
    #         "neg_parent_rel_corr_h_input_ids": negative_parent_relation_corrupted_heads_input_ids,
    #         "neg_parent_rel_corr_h_att_mask": negative_parent_relation_corrupted_heads_att_mask,
    #         "neg_parent_rel_corr_t_input_ids": negative_parent_relation_corrupted_tails_input_ids,
    #         "neg_parent_rel_corr_t_att_mask": negative_parent_relation_corrupted_tails_att_mask,
    #
    #         "pos_child_input_ids": positive_child_of_parent_input_ids,
    #         "pos_child_att_mask": positive_child_of_parent_att_mask,
    #         "neg_child_rel_corr_h_input_ids": negative_child_relation_corrupted_heads_input_ids,
    #         "neg_child_rel_corr_h_att_mask": negative_child_relation_corrupted_heads_att_mask,
    #         "neg_child_rel_corr_t_input_ids": negative_child_relation_corrupted_tails_input_ids,
    #         "neg_child_rel_corr_t_att_mask": negative_child_relation_corrupted_tails_att_mask,
    #
    #     }
    #     return sample


class HeterogeneousPositivePairNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], node_id2sem_group: Dict[int, str],
                 term_id2tokenizer_output: Dict, rel_ids, node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(HeterogeneousPositivePairNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id2token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list
        self.node_id2sem_group = node_id2sem_group
        self.rel_ids = rel_ids
        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

        self.num_edges = self.edge_index.size()[1]

        assert self.num_edges == len(rel_ids)

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])
        assert len(triplet_concept_ids) == len(term_1_input_ids)

        (batch_size, n_id, adjs) = super(HeterogeneousPositivePairNeighborSampler, self).sample(triplet_concept_ids)
        sem_groups = [self.node_id2sem_group[i.item()] for i in n_id]
        neighbor_node_ids = n_id[batch_size:]

        if isinstance(adjs, list):
            adj = adjs[0]
        else:
            adj = adjs
        edge_index = adj.edge_index
        src_semantic_groups, trg_semantic_groups = self.get_node_semantic_groups(edge_index=edge_index,
                                                                                 node_ids=n_id)
        # src_semantic_groups = torch.LongTensor(src_semantic_groups)
        # trg_semantic_groups = torch.LongTensor(trg_semantic_groups)

        e_ids_list = adj.e_id
        rel_ids_list = [self.rel_ids[e_id] for e_id in e_ids_list]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "edge_index": edge_index,
            "src_semantic_groups": src_semantic_groups, "trg_semantic_groups": trg_semantic_groups,
            "batch_size": batch_size, "concept_ids": triplet_concept_ids, "rel_ids_list": rel_ids_list,
            "sem_groups": sem_groups
        }
        return batch_dict

    def get_node_semantic_groups(self, edge_index, node_ids, ) -> Tuple[List[str], List[str]]:

        num_batch_nodes = edge_index.size(1)
        src_sem_groups_list = []
        trg_sem_groups_list = []
        for i in range(num_batch_nodes):
            src_batch_node_id, trg_batch_node_id = edge_index[0][i], edge_index[1][i]
            src_global_node_id, trg_global_node_id = node_ids[src_batch_node_id], node_ids[trg_batch_node_id]
            src_sem_group = self.node_id2sem_group[src_global_node_id.item()]
            trg_sem_group = self.node_id2sem_group[trg_global_node_id.item()]

            src_sem_groups_list.append(src_sem_group)
            trg_sem_groups_list.append(trg_sem_group)

        return src_sem_groups_list, trg_sem_groups_list


def graph_to_hetero_dataset(edge_index, hetero_dataset, all_node_types, sem_group_rel_combs, sem_groups,
                            node_features, src_node_sem_groups, trg_node_sem_groups,
                            rel_types, emb_size):
    # hetero_dataset = HeteroData()
    unique_nodes_grouped_by_sem_type = {}
    node2id_grouped_by_sem_group = {}
    num_batch_nodes = edge_index.size(1)

    for node_type in all_node_types:
        hetero_dataset[node_type].x = torch.zeros((1, emb_size), dtype=torch.float32)
    for (node_type_1, node_type_2, rel_id) in sem_group_rel_combs:
        hetero_dataset[node_type_1, str(rel_id), node_type_2].edge_index = torch.zeros((2, 0), dtype=torch.long)

    for i in range(num_batch_nodes):
        src_node_id, trg_node_id = edge_index[0][i], edge_index[1][i]
        src_sem_group, trg_sem_group = src_node_sem_groups[i], trg_node_sem_groups[i]
        if unique_nodes_grouped_by_sem_type.get(src_sem_group) is None:
            unique_nodes_grouped_by_sem_type[src_sem_group] = set()
        if unique_nodes_grouped_by_sem_type.get(trg_sem_group) is None:
            unique_nodes_grouped_by_sem_type[trg_sem_group] = set()

        unique_nodes_grouped_by_sem_type[src_sem_group].add(src_node_id.item())
        unique_nodes_grouped_by_sem_type[trg_sem_group].add(trg_node_id.item())
    for n_id, sem_gr in enumerate(sem_groups):
        if unique_nodes_grouped_by_sem_type.get(sem_gr) is None:
            unique_nodes_grouped_by_sem_type[sem_gr] = set()
        unique_nodes_grouped_by_sem_type[sem_gr].add(n_id)

    for sem_gr, sem_gr_node_ids in unique_nodes_grouped_by_sem_type.items():
        node2id_grouped_by_sem_group[sem_gr] = {orig_node_id: i for i, orig_node_id in enumerate(sem_gr_node_ids)}
    local_node_id2batch_node_id_grouped_by_sem_group = {}
    for sem_gr in node2id_grouped_by_sem_group.keys():
        ts = [(orig_node_id, i) for orig_node_id, i in node2id_grouped_by_sem_group[sem_gr].items()]
        ts.sort(key=lambda t: t[1])
        feature_ind = [t[0] for t in ts]
        hetero_dataset[sem_gr].x = node_features[feature_ind, :]
        local_node_id2batch_node_id_grouped_by_sem_group[sem_gr] = \
            {i: orig_node_id for (orig_node_id, i) in node2id_grouped_by_sem_group[sem_gr].items()}

    edge_index_dict_tmp = {}
    for i in range(num_batch_nodes):
        src_global_node_id, trg_global_node_id, rel_id = edge_index[0][i], edge_index[1][i], rel_types[i]
        src_sem_group, trg_sem_group = src_node_sem_groups[i], trg_node_sem_groups[i]

        assert ('|' not in src_sem_group) and ('|' not in str(rel_id)) and ('|' not in trg_sem_group)
        edge_str = f"{src_sem_group}|{rel_id.item()}|{trg_sem_group}"

        src_local_node_id = node2id_grouped_by_sem_group[src_sem_group][src_global_node_id.item()]
        trg_local_node_id = node2id_grouped_by_sem_group[trg_sem_group][trg_global_node_id.item()]
        if edge_index_dict_tmp.get(edge_str) is None:
            edge_index_dict_tmp[edge_str] = []

        edge_index_dict_tmp[edge_str].append((src_local_node_id, trg_local_node_id))
    # edge_index_dict = {}
    for edge_str, edge_tuples in edge_index_dict_tmp.items():
        attrs = edge_str.split('|')
        src_sem_group, rel_id, trg_sem_group = attrs[0], attrs[1], attrs[2]
        num_rels = len(edge_tuples)
        e_index = torch.zeros(size=(2, num_rels), dtype=torch.long)
        for i in range(num_rels):
            (src_node_id, trg_node_id) = edge_tuples[i]
            e_index[0][i] = src_node_id
            e_index[1][i] = trg_node_id
        hetero_dataset[src_sem_group, rel_id, trg_sem_group].edge_index = e_index

    return hetero_dataset, local_node_id2batch_node_id_grouped_by_sem_group


class HeterogeneousPositivePairNeighborSamplerV2(HGTLoader):
    def __init__(self, pos_pairs_term_1_id_list: List[int],
                 pos_pairs_term_2_id_list: List[int], pos_pairs_concept_ids_list: List[int],
                 node_id2sem_group: Dict[int, str], edge_index: torch.Tensor, emb_size:int,
                 term_id2tokenizer_output: Dict, rel_ids, node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        num_nodes = len(node_id2token_ids_dict)
        self.num_pos_pairs = len(pos_pairs_term_1_id_list)
        pos_pairs_index = np.arange(self.num_pos_pairs)
        pos_pairs_index = torch.from_numpy(pos_pairs_index)
        self.num_nodes = num_nodes

        self.node_id2token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list
        self.node_id2sem_group = node_id2sem_group

        self.rel_ids = rel_ids
        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length
        self.edge_index = edge_index
        self.num_edges = self.edge_index.size()[1]
        self.emb_size = emb_size

        assert self.num_edges == len(rel_ids)
        hetero_dataset = self.create_hetero_dataset(num_nodes=num_nodes, edge_index=edge_index,
                                                    edge_rel_ids=rel_ids)

        self.hetero_dataset = hetero_dataset
        super(HeterogeneousPositivePairNeighborSamplerV2, self).__init__(data=hetero_dataset,
                                                                         input_nodes=("SRC", pos_pairs_index),
                                                                         *args, **kwargs)

    def create_hetero_dataset(self, num_nodes: int, edge_index: torch.LongTensor, edge_rel_ids) -> HeteroData:
        hetero_dataset = HeteroData()

        num_edges = edge_index.size(1)
        logging.info(f"Creating heterogeneous dataset for {num_nodes} nodes and {num_edges} edges ...")
        assert num_edges == len(edge_rel_ids)
        # Grouping node ids by semantic group
        sem_group2global_node_ids = {}
        for i in range(num_edges):
            src_node_id = edge_index[0][i].item()
            src_sem_gr = self.node_id2sem_group[src_node_id]
            if sem_group2global_node_ids.get(src_sem_gr) is None:
                sem_group2global_node_ids[src_sem_gr] = set()
            sem_group2global_node_ids[src_sem_gr].add(src_node_id)
        for node_id in range(num_nodes):
            sem_gr = self.node_id2sem_group[node_id]
            if sem_group2global_node_ids.get(sem_gr) is None:
                sem_group2global_node_ids[sem_gr] = set()
            sem_group2global_node_ids[sem_gr].add(node_id)
        for sem_gr in sem_group2global_node_ids.keys():
            sem_group2global_node_ids[sem_gr] = sorted(sem_group2global_node_ids[sem_gr])
        t = list(range(num_nodes))
        # t = torch.from_numpy(t).unsqueeze(1)
        sem_group2global_node_ids["SRC"] = t

        # Creating a mapping from a global node id to a local node id of the given semantic group
        node_id2sem_group_node_id = {}
        for sem_gr, node_ids in sem_group2global_node_ids.items():
            # node_ids.sort()
            node_id2sem_group_node_id[sem_gr] = \
                {global_node_id: sem_gr_node_id for sem_gr_node_id, global_node_id in enumerate(node_ids)}
            node_ids = torch.LongTensor(node_ids).unsqueeze(1)
            # A feature of semantic group's node is the global node id of this node
            hetero_dataset[sem_gr].x = node_ids
            num_sem_group_nodes = node_ids.size(0)
            edge_index_range = torch.arange(num_sem_group_nodes, dtype=torch.long)
            self_loop_edge_index = edge_index_range.repeat(2, 1)
            hetero_dataset[sem_gr, "SELF-LOOP", sem_gr].edge_index = self_loop_edge_index
        # All source nodes have the same type "SRC"
        t = np.arange(num_nodes)
        t = torch.from_numpy(t).unsqueeze(1)
        hetero_dataset["SRC"].x = t


        sem_group_rel_dict = {}
        logging.info("Processing edges for heterogeneous dataset")
        for e_id in tqdm(range(num_edges), miniters=num_edges // 100):
            # Getting global node id of each edge
            global_src_node_id = edge_index[0][e_id].item()
            global_trg_node_id = edge_index[1][e_id].item()
            # Getting relation (REL) id of an edge
            rel_id = edge_rel_ids[e_id]
            # Given a global node id, we can determine a semantic group of this node
            src_sem_group = self.node_id2sem_group[global_src_node_id]
            trg_sem_group = self.node_id2sem_group[global_trg_node_id]
            # Given a semantic group and global node id, we can determine a local node id in this semantic group
            local_src_node_id = node_id2sem_group_node_id[src_sem_group][global_src_node_id]
            local_trg_node_id = node_id2sem_group_node_id[trg_sem_group][global_trg_node_id]
            assert local_src_node_id < hetero_dataset[src_sem_group].x.size(0)
            assert local_trg_node_id < hetero_dataset[trg_sem_group].x.size(0)
            # First, group all edges by <sem group, relation, sem group> with no repetition
            sem_group_rel_combination_str = f"{src_sem_group}|{rel_id}|SRC"
            if sem_group_rel_dict.get(sem_group_rel_combination_str) is None:
                sem_group_rel_dict[sem_group_rel_combination_str] = set()
            sem_group_rel_dict[sem_group_rel_combination_str].add(f"{local_src_node_id}|{global_trg_node_id}")
        # Second, for each unique <sem group, relation, sem group> combination we create an edge index that
        # contains local source and target node ids
        logging.info(f"Creating edge index for each of {len(sem_group_rel_dict)} "
                     f"<sem group, relation, sem group> combination")
        for sem_group_rel_combination_str, edge_strs_set in tqdm(sem_group_rel_dict.items()):
            (src_sem_group, rel_id, trg_sem_group) = sem_group_rel_combination_str.split('|')
            num_sem_gr_rel_comb_edges = len(edge_strs_set)
            sem_gr_rel_comb_edge_index = torch.zeros(size=(2, num_sem_gr_rel_comb_edges), dtype=torch.long)
            for e_id, edge_str in enumerate(edge_strs_set):
                (local_src_node_id, local_trg_node_id) = map(int, edge_str.split('|'))
                sem_gr_rel_comb_edge_index[0][e_id] = local_src_node_id
                sem_gr_rel_comb_edge_index[1][e_id] = local_trg_node_id
            hetero_dataset[src_sem_group, rel_id, trg_sem_group].edge_index = sem_gr_rel_comb_edge_index
        logging.info(f"Finished creating heterogeneous dataset.")

        return hetero_dataset

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size


    def transform_fn(self, batch: Any) -> HeteroData:

        node_dict, row_dict, col_dict, edge_dict, batch_size = batch["hetero_dataset"]
        data = self.filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                  edge_dict, self.perm_dict)

        data[self.input_nodes[0]].batch_size = batch_size

        batch["hetero_dataset"] = self.transform(data) if self.transform is not None else data

        return batch

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])
        assert len(triplet_concept_ids) == len(term_1_input_ids)

        hetero_sub_dataset = super(HeterogeneousPositivePairNeighborSamplerV2, self).sample(triplet_concept_ids)

        node_dict, row_dict, col_dict, edge_dict, batch_size = hetero_sub_dataset

        hetero_sub_dataset_x_dict = node_dict

        edge_index_dict = {}
        for key in row_dict.keys():
            edge_index_row = row_dict[key]
            edge_index_col = col_dict[key]

            assert len(edge_index_row.size()) == len(edge_index_col.size()) == 1
            edge_index = torch.LongTensor(torch.stack((edge_index_row.long(), edge_index_col.long())))

            edge_index_dict[key] = edge_index

        hetero_sub_dataset_bert_input_1 = {}
        hetero_sub_dataset_bert_input_2 = {}
        for node_type, node_ids in hetero_sub_dataset_x_dict.items():
            neighbor_input_ids_1, neighbor_att_masks_1 = node_ids2tokenizer_output(
                batch=node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
                seq_max_length=self.seq_max_length)
            neighbor_input_ids_2, neighbor_att_masks_2 = node_ids2tokenizer_output(
                batch=node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
                seq_max_length=self.seq_max_length)
            neighbor_input_1 = (neighbor_input_ids_1, neighbor_att_masks_1)
            neighbor_input_2 = (neighbor_input_ids_2, neighbor_att_masks_2)
            assert neighbor_input_ids_1.size() == neighbor_att_masks_1.size()
            assert neighbor_input_ids_2.size() == neighbor_att_masks_2.size()

            hetero_sub_dataset_bert_input_1[node_type] = neighbor_input_1
            hetero_sub_dataset_bert_input_2[node_type] = neighbor_input_2

        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "concept_ids": triplet_concept_ids,
            "hetero_dataset": hetero_sub_dataset, "nodes_bert_input_1": hetero_sub_dataset_bert_input_1,
            "nodes_bert_input_2": hetero_sub_dataset_bert_input_2, "batch_size": batch_size,
        }

        return batch_dict

    def filter_hetero_data(self,
                           data: HeteroData,
                           node_dict: Dict[str, Tensor],
                           row_dict: Dict[str, Tensor],
                           col_dict: Dict[str, Tensor],
                           edge_dict: Dict[str, Tensor],
                           perm_dict: Dict[str, OptTensor],
                           ) -> HeteroData:
        # Filters a heterogeneous data object to only hold nodes in `node` and
        # edges in `edge` for each node and edge type, respectively:
        out = HeteroData() # copy.copy(data)

        for node_type in data.node_types:

            if node_dict.get(node_type) is not None:
                filter_node_store_(data[node_type], out[node_type],
                                   node_dict[node_type])
            else:
                out[node_type].x = torch.zeros(size=(1, self.emb_size), dtype=torch.float)

        for edge_type in data.edge_types:
            edge_type_str = edge_type_to_str(edge_type)

            if row_dict.get(edge_type_str) is None:
                out[edge_type].edge_index = torch.zeros(size=(2, 1), dtype=torch.long)
            else:
                self.filter_edge_store_(data[edge_type], out[edge_type],
                                        row_dict[edge_type_str], col_dict[edge_type_str],
                                        edge_dict[edge_type_str], perm_dict[edge_type_str])


        return out

    @staticmethod
    def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                           col: Tensor, index: Tensor,
                           perm: OptTensor = None) -> EdgeStorage:
        # Filters a edge storage object to only hold the edges in `index`,
        # which represents the new graph as denoted by `(row, col)`:
        for key, value in store.items():
            if key == 'edge_index':
                edge_index = torch.stack([row, col], dim=0)
                out_store.edge_index = edge_index.to(value.device)

            elif key == 'adj_t':
                # NOTE: We expect `(row, col)` to be sorted by `col` (CSC layout).
                row = row.to(value.device())
                col = col.to(value.device())
                edge_attr = value.storage.value()
                if edge_attr is not None:
                    index = index.to(edge_attr.device)
                    edge_attr = edge_attr[index]
                sparse_sizes = store.size()[::-1]
                out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                               sparse_sizes=sparse_sizes,
                                               is_sorted=True)

            elif store.is_edge_attr(key):
                if perm is None:
                    index = index.to(value.device)
                    out_store[key] = index_select(value, index, dim=0)
                else:
                    perm = perm.to(value.device)
                    index = index.to(value.device)
                    out_store[key] = index_select(value, perm[index], dim=0)

        return out_store
