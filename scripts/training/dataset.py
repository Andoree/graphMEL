import logging
import random
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from transformers import AutoTokenizer, AutoModel

from graphmel.scripts.utils.io import load_node_id2terms_list, load_edges_tuples


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
    logging.info("Converting edge tuples to edge index")
    edge_index = torch.zeros(size=[2, len(edges_tuples)], dtype=torch.long)
    for idx, (id_1, id_2) in enumerate(edges_tuples):
        edge_index[0][idx] = id_1
        edge_index[1][idx] = id_2
    logging.info(f"Edge index is created. The size is {edge_index.size()}, there are {edge_index.max()} nodes")

    return edge_index


class Node2vecDataset(Dataset):
    def __init__(self, edge_index, node_id_to_token_ids_dict: Dict[int, List[List[int]]], walk_length: int,
                 walks_per_node: int, p: float, q: float, num_negative_samples: int, context_size: int,
                 seq_max_length, num_nodes=None, ):
        assert walk_length >= context_size
        self.node_id_to_token_ids_dict = node_id_to_token_ids_dict
        self.walks_per_node = walks_per_node
        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to('cpu')
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.seq_max_length = seq_max_length

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.adj.sparse_size(0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        pos_batch = self.pos_sample(batch)
        neg_batch = self.neg_sample(batch)

        pos_batch_input_ids, pos_batch_attention_masks = self.node_ids2tokenizer_output(pos_batch)
        neg_batch_input_ids, neg_batch_attention_masks = self.node_ids2tokenizer_output(neg_batch)

        return pos_batch_input_ids, pos_batch_attention_masks, neg_batch_input_ids, neg_batch_attention_masks

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)

        rowptr, col, _ = self.adj.coo()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.adj.sparse_size(0),
                           (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def node_ids2tokenizer_output(self, batch):
        batch_size = batch.size()[0]
        num_samples = batch.size()[1]

        tokenizer_outputs = [random.choice(self.node_id_to_token_ids_dict[node_id.item()]) for node_id in
                             batch.view(-1)]
        batch_input_ids = torch.stack([tok_output["input_ids"][0] for tok_output in tokenizer_outputs])
        batch_attention_masks = torch.stack([tok_output["attention_mask"][0] for tok_output in tokenizer_outputs])

        batch_input_ids = batch_input_ids.view(batch_size, num_samples, self.seq_max_length)
        batch_attention_masks = batch_attention_masks.view(batch_size, num_samples, self.seq_max_length)

        return batch_input_ids, batch_attention_masks


def load_data_and_bert_model(train_node2terms_path: str, train_edges_path: str, val_node2terms_path: str,
                             val_edges_path: str, text_encoder_name: str, text_encoder_seq_length: int,
                             drop_relations_info):
    train_node_id2terms_dict = load_node_id2terms_list(dict_path=train_node2terms_path, )
    train_edges_tuples = load_edges_tuples(train_edges_path)
    if drop_relations_info:
        train_edges_tuples = [(t[0], t[1]) for t in train_edges_tuples]
    val_node_id2terms_dict = load_node_id2terms_list(dict_path=val_node2terms_path, )
    val_edges_tuples = load_edges_tuples(val_edges_path)
    if drop_relations_info:
        val_edges_tuples = [(t[0], t[1]) for t in val_edges_tuples]
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    bert_encoder = AutoModel.from_pretrained(text_encoder_name)
    train_node_id2token_ids_dict = tokenize_node_terms(train_node_id2terms_dict, tokenizer,
                                                       max_length=text_encoder_seq_length)
    # train_num_nodes = len(set(train_node_id2terms_dict.keys()))
    train_edge_index = convert_edges_tuples_to_edge_index(edges_tuples=train_edges_tuples)
    val_node_id2token_ids_dict = tokenize_node_terms(val_node_id2terms_dict, tokenizer,
                                                     max_length=text_encoder_seq_length)
    # val_num_nodes = len(set(val_node_id2terms_dict.keys()))
    val_edge_index = convert_edges_tuples_to_edge_index(edges_tuples=val_edges_tuples)

    return bert_encoder, train_node_id2token_ids_dict, train_edge_index, val_node_id2token_ids_dict, val_edge_index
