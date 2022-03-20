import logging
import os
import random
from argparse import ArgumentParser

import numpy as np

from graphmel.scripts.training.dataset import tokenize_node_terms, NeighborSampler, convert_edges_tuples_to_edge_index
from graphmel.scripts.training.model import GraphSAGEOverBert
from graphmel.scripts.utils.io import load_node_id2terms_list, load_tuples, update_log_file
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import NeighborSampler as RawNeighborSampler
from transformers import AutoTokenizer
from transformers import AutoModel
from torch_geometric.nn import Node2Vec


def node2vec_train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in train_loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def node2vec_val_epoch(model,):
    with torch.no_grad():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=10)
    return acc


for epoch in range(1, 101):
    loss = train()
    #acc = test()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


# TODO: Добавить логирование кривых обучения по эпохам
# TODO: Откусить испанский (или русский) для того, чтобы произвести на нём отладку
# TODO: Рисовать кривые обучения, надо же как-то затюнить параметры
def main(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    node_id2terms_dict = load_node_id2terms_list(dict_path=args.node2terms_path, )
    edges_tuples = load_tuples(args.train_edges_path)

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    bert_encoder = AutoModel.from_pretrained(args.text_encoder)

    node_id2token_ids_dict = tokenize_node_terms(node_id2terms_dict, tokenizer, max_length=args.text_encoder_seq_length)
    num_nodes = len(set(node_id2terms_dict.keys()))
    edge_index = convert_edges_tuples_to_edge_index(edges_tuples=edges_tuples)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nodes = edge_index.t().numpy()
    nodes = np.unique(list(nodes[:, 0]) + list(nodes[:, 1]))
    np.random.shuffle(nodes)
    logging.info(f"Overall number of nodes: {len(nodes)}")

    train_size = int(len(nodes) * args.train_proportion)
    train_set = nodes[:train_size]
    val_set = nodes[train_size:]

    train_mask = torch.zeros(len(nodes), dtype=torch.long,)
    for i in train_set:
        train_mask[i] = 1.
    val_mask = torch.zeros(len(nodes), dtype=torch.long,)
    for i in val_set:
        val_mask[i] = 1.

    model = Node2Vec(edge_index, embedding_dim=args.node2vec_emb_dim, walk_length=args.random_walk_length,
                     context_size=args.node2vec_context_size, walks_per_node=args.node2vec_walks_per_node,
                     num_negative_samples=args.node2vec_num_negative_samples, p=args.node2vec_p, q=args.node2vec_q,
                     sparse=True).to(device)
    loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.learning_rate)

    if args.gpus > 1:
        model = nn.DataParallel(model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--node2terms_path', type=str)
    parser.add_argument('--edges_path', type=str)
    parser.add_argument('--train_proportion', type=float)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--text_encoder_seq_length', type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--graph_num_neighbors', type=int, nargs='+', )
    parser.add_argument('--node2vec_emb_dim', type=int, required=False)
    parser.add_argument('--node2vec_num_negative_samples', type=int, required=False)
    parser.add_argument('--node2vec_context_size', type=int, required=False)
    parser.add_argument('--node2vec_walks_per_node', type=int, required=False)
    parser.add_argument('--node2vec_p', type=int, required=False)
    parser.add_argument('--node2vec_q', type=int, required=False)
    parser.add_argument('--random_walk_length', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--random_state', type=int)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()
    seed = arguments.random_state
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    main(arguments)
