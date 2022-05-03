import logging
import random
from typing import Dict

import torch
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_sparse import SparseTensor

from graphmel.scripts.training.data.data_utils import node_ids2tokenizer_output


class RelationalNeighborSampler(RawNeighborSampler):
    def __init__(self, edge_index, node_neighborhood_sizes, rel_ids, num_nodes: int, node_id_to_token_ids_dict,
                 seq_max_length: int,
                 rel_id2inverse_rel_id: Dict[int, int], *args, **kwargs):
        super(RelationalNeighborSampler, self).__init__(edge_index=edge_index, sizes=node_neighborhood_sizes, *args,
                                                        **kwargs)
        self.batch_size = kwargs["batch_size"]
        self.edge_index = edge_index
        self.random_walk_length = node_neighborhood_sizes
        self.rel_ids = rel_ids
        self.sizes = node_neighborhood_sizes
        self.num_edges = edge_index.size()[1]
        self.num_nodes = num_nodes
        self.node_id_to_token_ids_dict = node_id_to_token_ids_dict
        self.seq_max_length = seq_max_length
        self.rel_id2inverse_rel_id = rel_id2inverse_rel_id
        self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  value=rel_ids, sparse_sizes=(num_nodes, num_nodes)).t()
        assert self.num_edges == len(rel_ids)
        # Constructing a set of unique edge identifiers for fast edge existence check during negative edge sampling
        self.unique_edges_strings = set()
        # Filtering unique tuples (src_node, trg_node, rel, rela)
        logging.info(f"Dataset creation: creating set of unique edges, {edge_index.size()[1]} edges with duplicates")
        for edge_id, rel_id in zip(range(self.num_edges), rel_ids):
            node_id_1 = self.edge_index[0][edge_id]
            node_id_2 = self.edge_index[1][edge_id]
            edge_s = f"{node_id_1}~{node_id_2}~{rel_id}"
            self.unique_edges_strings.add(edge_s)
        logging.info(f"Dataset is created, there are {len(self.unique_edges_strings)} unique edges strings")

    def __len__(self):
        return self.num_edges

    # TODO: В статье есть петля? W_o
    def sample(self, edge_ids):

        batch_size = len(edge_ids)
        rel_ids = self.rel_ids[edge_ids]
        pos_batch = {
            "source_node_id": self.edge_index[0][edge_ids],
            "target_node_id": self.edge_index[1][edge_ids],
            "rel_id": rel_ids,
        }
        # Creating negative samples by corrupting either head or tail of each edge
        neg_batch = self.neg_sample(pos_batch)
        pos_src_node_ids = pos_batch["source_node_id"]
        pos_trg_node_ids = pos_batch["target_node_id"]
        neg_src_node_ids = neg_batch["source_node_id"]
        neg_trg_node_ids = neg_batch["target_node_id"]

        (pos_src_batch_size, pos_src_n_id, pos_src_adjs) = super(RelationalNeighborSampler, self).sample(
            pos_src_node_ids)
        (pos_trg_batch_size, pos_trg_n_id, pos_trg_adjs) = super(RelationalNeighborSampler, self).sample(
            pos_trg_node_ids)
        (neg_src_batch_size, neg_src_n_id, neg_src_adjs) = super(RelationalNeighborSampler, self).sample(
            neg_src_node_ids)
        (neg_trg_batch_size, neg_trg_n_id, neg_trg_adjs) = super(RelationalNeighborSampler, self).sample(
            neg_trg_node_ids)
        # Getting BERT input for heads and tails of positive and negative edges
        pos_src_input_ids, pos_src_attention_masks = node_ids2tokenizer_output(batch=pos_src_n_id,
                                                                               seq_max_length=self.seq_max_length,
                                                                               node_id_to_token_ids_dict=self.node_id_to_token_ids_dict, )
        pos_trg_input_ids, pos_trg_attention_masks = node_ids2tokenizer_output(batch=pos_trg_n_id,
                                                                               seq_max_length=self.seq_max_length,
                                                                               node_id_to_token_ids_dict=self.node_id_to_token_ids_dict, )
        neg_src_input_ids, neg_src_attention_masks = node_ids2tokenizer_output(batch=neg_src_n_id,
                                                                               seq_max_length=self.seq_max_length,
                                                                               node_id_to_token_ids_dict=self.node_id_to_token_ids_dict, )
        neg_trg_input_ids, neg_trg_attention_masks = node_ids2tokenizer_output(batch=neg_trg_n_id,
                                                                               seq_max_length=self.seq_max_length,
                                                                               node_id_to_token_ids_dict=self.node_id_to_token_ids_dict, )
        assert pos_src_input_ids.size() == pos_src_attention_masks.size()
        assert pos_trg_input_ids.size() == pos_trg_attention_masks.size()
        assert neg_src_input_ids.size() == neg_src_attention_masks.size()
        assert neg_trg_input_ids.size() == neg_trg_attention_masks.size()

        pos_src_rel_ids = pos_src_adjs.e_id
        pos_trg_rel_ids = torch.LongTensor([self.rel_id2inverse_rel_id[int(idx)] for idx in pos_trg_adjs.e_id])
        neg_src_rel_ids = neg_src_adjs.e_id
        neg_trg_rel_ids = torch.LongTensor([self.rel_id2inverse_rel_id[int(idx)] for idx in neg_trg_adjs.e_id])

        inv_rel_ids = torch.LongTensor([self.rel_id2inverse_rel_id[int(idx)] for idx in self.rel_ids[edge_ids]])
        pos_src_input = (
            pos_src_batch_size, pos_src_adjs.edge_index, pos_src_input_ids, pos_src_attention_masks, pos_src_rel_ids)
        pos_trg_input = (
            pos_trg_batch_size, pos_trg_adjs.edge_index, pos_trg_input_ids, pos_trg_attention_masks, pos_trg_rel_ids)
        neg_src_input = (
            neg_src_batch_size, neg_src_adjs.edge_index, neg_src_input_ids, neg_src_attention_masks, neg_src_rel_ids)
        neg_trg_input = (
            neg_trg_batch_size, neg_trg_adjs.edge_index, neg_trg_input_ids, neg_trg_attention_masks, neg_trg_rel_ids)

        batch_dict = {
            "pos_src_input": pos_src_input, "pos_trg_input": pos_trg_input,
            "neg_src_input": neg_src_input, "neg_trg_input": neg_trg_input,
            "rel_id": rel_ids, "inv_rel_id": inv_rel_ids, "batch_size": batch_size
        }

        return batch_dict

    def neg_sample(self, pos_batch):
        """
        :param pos_batch: Positive batch's dictionary that contains 3 tensors:
            1. Head node's id of edge
            2. Tail node's id of edge
            3. Relation id
        :return: Negative batch that contains corrupted edges that are
        obtained by corrupting either head or tail of each edge
        """
        pos_source_node_ids = pos_batch["source_node_id"]
        pos_target_node_ids = pos_batch["target_node_id"]
        pos_rel_ids = pos_batch["rel_id"]
        neg_source_node_ids = []
        neg_target_node_ids = []
        for i, (source_id, target_id, rel_id) in enumerate(
                zip(pos_source_node_ids, pos_target_node_ids, pos_rel_ids)):
            p = random.uniform(0, 1)
            res_source_id = source_id
            res_target_id = target_id

            edge_str = f"{res_source_id}~{res_target_id}~{rel_id}"
            while edge_str in self.unique_edges_strings:
                res_source_id = source_id
                res_target_id = target_id
                if p < 0.5:
                    res_source_id = random.randint(0, self.num_nodes - 1)
                else:
                    res_target_id = random.randint(0, self.num_nodes - 1)
                edge_str = f"{res_source_id}~{res_target_id}~{rel_id}"
            neg_source_node_ids.append(res_source_id)
            neg_target_node_ids.append(res_target_id)
        neg_source_node_ids = torch.LongTensor(neg_source_node_ids)
        neg_target_node_ids = torch.LongTensor(neg_target_node_ids)
        neg_batch_dict = {
            "source_node_id": neg_source_node_ids,
            "target_node_id": neg_target_node_ids,
        }
        return neg_batch_dict
