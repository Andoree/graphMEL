import random
from typing import List, Dict
import logging
from torch.utils.data import Dataset
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
import torch
import numpy as np
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
        else:
            adjs = [adj for adj in adjs]

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
        assert len(concept_id2parents) == len(concept_id2childs)
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
        # children_of_parent_list = self.concept_id2children.get()
        # concept_children_list = self.concept_id2children.get(anchor_concept_id)

        # 0 - id of relation from parent to child, 1 - from child to parent
        if concept_parents_list is not None and len(concept_parents_list) > 0:
            positive_parent_concept_id = random.choice(concept_parents_list)
            # children_of_parent_list = self.concept_id2children.get(positive_parent_concept_id)
            # pos_relation: (positive_parent_concept_id, IS_A_CHILD_OF, concept_id)
            hierarchical_sample_weight = torch.FloatTensor([1, ])

            negative_parent_relation_corrupted_heads = \
                self.create_negative_samples(concept_id_to_remain=anchor_concept_id,
                                             concept_id2neighbors=self.concept_id2parents)
            negative_parent_relation_corrupted_tails = \
                self.create_negative_samples(concept_id_to_remain=positive_parent_concept_id,
                                             concept_id2neighbors=self.concept_id2children)

            positive_parent_tokenized = random.choice(self.node_id2token_ids_dict[positive_parent_concept_id])

            negative_parent_relation_corrupted_heads_tokenized = \
                [random.choice(self.node_id2token_ids_dict[i]) for i in negative_parent_relation_corrupted_heads]
            negative_parent_relation_corrupted_tails_tokenized = \
                [random.choice(self.node_id2token_ids_dict[i]) for i in negative_parent_relation_corrupted_tails]

            positive_parent_input_ids = positive_parent_tokenized["input_ids"][0]
            positive_parent_att_mask = positive_parent_tokenized["attention_mask"][0]
            negative_parent_relation_corrupted_heads_input_ids = torch.stack(
                [d["input_ids"][0] for d in negative_parent_relation_corrupted_heads_tokenized])
            negative_parent_relation_corrupted_heads_att_mask = torch.stack(
                [d["attention_mask"][0] for d in negative_parent_relation_corrupted_heads_tokenized])

            negative_parent_relation_corrupted_tails_input_ids = torch.stack(
                [d["input_ids"][0] for d in negative_parent_relation_corrupted_tails_tokenized])
            negative_parent_relation_corrupted_tails_att_mask = torch.stack(
                [d["attention_mask"][0] for d in negative_parent_relation_corrupted_tails_tokenized])

            # Child to parent relation sampling
            children_of_parent_list = self.concept_id2children.get(positive_parent_concept_id)
            positive_child_of_parent_concept_id = random.choice(children_of_parent_list)

            # pos_relation: (positive_child_of_parent_concept_id, IS_A_PARENT_OF, positive_parent_concept_id)
            negative_child_relation_corrupted_heads = \
                self.create_negative_samples(concept_id_to_remain=positive_parent_concept_id,
                                             concept_id2neighbors=self.concept_id2children)
            negative_child_relation_corrupted_tails = \
                self.create_negative_samples(concept_id_to_remain=positive_child_of_parent_concept_id,
                                             concept_id2neighbors=self.concept_id2parents)

            positive_child_of_parent_tokenized = \
                random.choice(self.node_id2token_ids_dict[positive_child_of_parent_concept_id])
            negative_child_relation_corrupted_heads_tokenized = \
                [random.choice(self.node_id2token_ids_dict[i]) for i in negative_child_relation_corrupted_heads]
            negative_child_relation_corrupted_tails_tokenized = \
                [random.choice(self.node_id2token_ids_dict[i]) for i in negative_child_relation_corrupted_tails]

            positive_child_of_parent_input_ids = positive_child_of_parent_tokenized["input_ids"][0]
            positive_child_of_parent_att_mask = positive_child_of_parent_tokenized["attention_mask"][0]

            negative_child_relation_corrupted_heads_input_ids = torch.stack(
                [d["input_ids"][0] for d in negative_child_relation_corrupted_heads_tokenized])
            negative_child_relation_corrupted_heads_att_mask = torch.stack(
                [d["attention_mask"][0] for d in negative_child_relation_corrupted_heads_tokenized])

            negative_child_relation_corrupted_tails_input_ids = torch.stack(
                [d["input_ids"][0] for d in negative_child_relation_corrupted_tails_tokenized])
            negative_child_relation_corrupted_tails_att_mask = torch.stack(
                [d["attention_mask"][0] for d in negative_child_relation_corrupted_tails_tokenized])

        else:
            hierarchical_sample_weight = torch.FloatTensor([0, ])
            positive_parent_input_ids = torch.zeros(self.seq_max_length, dtype=torch.long)
            positive_parent_att_mask = torch.zeros(self.seq_max_length, dtype=torch.long)
            negative_parent_relation_corrupted_heads_input_ids = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
            negative_parent_relation_corrupted_heads_att_mask = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
            negative_parent_relation_corrupted_tails_input_ids = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
            negative_parent_relation_corrupted_tails_att_mask = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)

            positive_child_of_parent_input_ids = torch.zeros(self.seq_max_length, dtype=torch.long)
            positive_child_of_parent_att_mask = torch.zeros(self.seq_max_length, dtype=torch.long)
            negative_child_relation_corrupted_heads_input_ids = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
            negative_child_relation_corrupted_heads_att_mask = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
            negative_child_relation_corrupted_tails_input_ids = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)
            negative_child_relation_corrupted_tails_att_mask = \
                torch.zeros((self.negative_sample_size, self.seq_max_length), dtype=torch.long)

        parent_rel_id = torch.LongTensor([0, ])
        child_rel_id = torch.LongTensor([1, ])

        anchor_concept_id = torch.LongTensor([anchor_concept_id, ])
        sample = {
            "term_1_input_ids": term_1_input_ids, "term_1_att_mask": term_1_att_mask,
            "term_2_input_ids": term_2_input_ids, "term_2_att_mask": term_2_att_mask,
            "anchor_concept_id": anchor_concept_id,
            "parent_rel_id": parent_rel_id, "child_rel_id": child_rel_id,

            "hierarchical_sample_weight": hierarchical_sample_weight,
            "pos_parent_input_ids": positive_parent_input_ids,
            "pos_parent_att_mask": positive_parent_att_mask,
            "neg_parent_rel_corr_h_input_ids": negative_parent_relation_corrupted_heads_input_ids,
            "neg_parent_rel_corr_h_att_mask": negative_parent_relation_corrupted_heads_att_mask,
            "neg_parent_rel_corr_t_input_ids": negative_parent_relation_corrupted_tails_input_ids,
            "neg_parent_rel_corr_t_att_mask": negative_parent_relation_corrupted_tails_att_mask,

            "pos_child_input_ids": positive_child_of_parent_input_ids,
            "pos_child_att_mask": positive_child_of_parent_att_mask,
            "neg_child_rel_corr_h_input_ids": negative_child_relation_corrupted_heads_input_ids,
            "neg_child_rel_corr_h_att_mask": negative_child_relation_corrupted_heads_att_mask,
            "neg_child_rel_corr_t_input_ids": negative_child_relation_corrupted_tails_input_ids,
            "neg_child_rel_corr_t_att_mask": negative_child_relation_corrupted_tails_att_mask,

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

        parent_rel_id = torch.stack([d["parent_rel_id"][0] for d in data], dim=0)
        child_rel_id = torch.stack([d["child_rel_id"][0] for d in data], dim=0)
        assert term_1_input_ids.size(0) == anchor_concept_id.size(0) == parent_rel_id.size(0) == child_rel_id.size(0)

        hierarchical_sample_weight = torch.stack([d["hierarchical_sample_weight"][0] for d in data], dim=0)
        # hierarchical_sample_weight = hierarchical_sample_weight > 0
        pos_parent_input_ids = torch.stack([d["pos_parent_input_ids"] for d in data], dim=0).unsqueeze(1)
        pos_parent_att_mask = torch.stack([d["pos_parent_att_mask"] for d in data], dim=0).unsqueeze(1)

        neg_parent_rel_corr_h_input_ids = torch.stack([d["neg_parent_rel_corr_h_input_ids"] for d in data], dim=0)
        neg_parent_rel_corr_h_att_mask = torch.stack([d["neg_parent_rel_corr_h_att_mask"] for d in data], dim=0)
        neg_parent_rel_corr_t_input_ids = torch.stack([d["neg_parent_rel_corr_t_input_ids"] for d in data], dim=0)
        neg_parent_rel_corr_t_att_mask = torch.stack([d["neg_parent_rel_corr_t_att_mask"] for d in data], dim=0)
        assert neg_parent_rel_corr_h_input_ids.size() == neg_parent_rel_corr_h_att_mask.size() \
               == neg_parent_rel_corr_t_input_ids.size() == neg_parent_rel_corr_t_att_mask.size()

        pos_child_input_ids = torch.stack([d["pos_child_input_ids"] for d in data], dim=0).unsqueeze(1)
        pos_child_att_mask = torch.stack([d["pos_child_att_mask"] for d in data], dim=0).unsqueeze(1)
        assert pos_parent_input_ids.size() == pos_parent_att_mask.size() \
               == pos_child_input_ids.size() == pos_child_att_mask.size()

        neg_child_rel_corr_h_input_ids = torch.stack([d["neg_child_rel_corr_h_input_ids"] for d in data], dim=0)
        neg_child_rel_corr_h_att_mask = torch.stack([d["neg_child_rel_corr_h_att_mask"] for d in data], dim=0)
        neg_child_rel_corr_t_input_ids = torch.stack([d["neg_child_rel_corr_t_input_ids"] for d in data], dim=0)
        neg_child_rel_corr_t_att_mask = torch.stack([d["neg_child_rel_corr_t_att_mask"] for d in data], dim=0)
        assert neg_child_rel_corr_h_input_ids.size() == neg_child_rel_corr_h_att_mask.size() \
               == neg_child_rel_corr_t_input_ids.size() == neg_child_rel_corr_t_att_mask.size()

        batch = {
            "term_1_input_ids": term_1_input_ids, "term_1_att_mask": term_1_att_mask,
            "term_2_input_ids": term_2_input_ids, "term_2_att_mask": term_2_att_mask,
            "anchor_concept_id": anchor_concept_id,
            "parent_rel_id": parent_rel_id, "child_rel_id": child_rel_id,

            "hierarchical_sample_weight": hierarchical_sample_weight,
            "pos_parent_input_ids": pos_parent_input_ids,
            "pos_parent_att_mask": pos_parent_att_mask,
            "neg_parent_rel_corr_h_input_ids": neg_parent_rel_corr_h_input_ids,
            "neg_parent_rel_corr_h_att_mask": neg_parent_rel_corr_h_att_mask,
            "neg_parent_rel_corr_t_input_ids": neg_parent_rel_corr_t_input_ids,
            "neg_parent_rel_corr_t_att_mask": neg_parent_rel_corr_t_att_mask,
            # "pos_child_mask": pos_child_mask,
            "pos_child_input_ids": pos_child_input_ids,
            "pos_child_att_mask": pos_child_att_mask,
            "neg_child_rel_corr_h_input_ids": neg_child_rel_corr_h_input_ids,
            "neg_child_rel_corr_h_att_mask": neg_child_rel_corr_h_att_mask,
            "neg_child_rel_corr_t_input_ids": neg_child_rel_corr_t_input_ids,
            "neg_child_rel_corr_t_att_mask": neg_child_rel_corr_t_att_mask,

        }

        return batch
