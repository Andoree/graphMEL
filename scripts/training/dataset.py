import random
from typing import Dict, List, Tuple
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
import torch


def tokenize_node_terms(node_id_to_terms_dict, tokenizer, max_length: int) -> Dict[int, List[List[int]]]:
    node_id_to_token_ids_dict = {}
    for node_id, terms_list in node_id_to_terms_dict.items():
        node_tokenized_terms = []
        for term in terms_list:
            tokenizer_output = tokenizer.encode_plus(term, max_length=max_length,
                                                     padding="max_length", truncation=True,
                                                     return_tensors="pt", )
            node_tokenized_terms.append(tokenizer_output)
        node_id_to_token_ids_dict[node_id] = node_tokenized_terms
    return node_id_to_token_ids_dict


class NeighborSampler(RawNeighborSampler):
    def __init__(self, node_id_to_token_ids_dict, seq_max_length, random_walk_length: int, *args, **kwargs):
        super(NeighborSampler, self).__init__(*args, **kwargs)
        self.node_id_to_token_ids_dict = node_id_to_token_ids_dict
        self.seq_max_length = seq_max_length
        self.random_walk_length = random_walk_length
        self.num_nodes = kwargs["num_nodes"]
        # TODO: Deterministic random_walk для валидации?

    def __len__(self):
        return self.num_nodes

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=self.random_walk_length, coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),), dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)

        (batch_size, n_id, adjs) = super(NeighborSampler, self).sample(batch)
        batch_input_ids = []
        batch_attention_masks = []
        for idx in n_id:
            idx = idx.item()
            tokenized_terms_list = self.node_id_to_token_ids_dict[idx]
            selected_term_tokenizer_output = random.choice(tokenized_terms_list)
            input_ids = selected_term_tokenizer_output["input_ids"][0]
            attention_mask = selected_term_tokenizer_output["attention_mask"][0]
            assert len(input_ids) == len(attention_mask) == self.seq_max_length
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)

        return batch_size, n_id, adjs, batch_input_ids, batch_attention_masks


def convert_edges_tuples_to_edge_index(edges_tuples: List[Tuple[int, int]]) -> torch.Tensor:
    edge_index = torch.zeros(size=[2, len(edges_tuples)], dtype=torch.long)
    for idx, (id_1, id_2) in enumerate(edges_tuples):
        edge_index[0][idx] = id_1
        edge_index[1][idx] = id_2

    return edge_index
