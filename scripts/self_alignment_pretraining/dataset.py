import random
from typing import List, Dict

from torch.utils.data import Dataset
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
import torch

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
        return len(self.triplet_first_term_id_list) // self.batch_size


    def sample(self, batch):
        term_1_ids = [self.triplet_first_term_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.triplet_second_term_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])
        assert len(triplet_concept_ids) == self.seq_max_length

        row, col, _ = self.adj_t.coo()

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

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=1)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=1)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=1)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=1)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": triplet_concept_ids
        }

        return batch_dict
