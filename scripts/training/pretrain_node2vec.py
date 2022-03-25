import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphmel.scripts.training.dataset import tokenize_node_terms, NeighborSampler, convert_edges_tuples_to_edge_index, \
    Node2vecDataset
from graphmel.scripts.training.model import GraphSAGEOverBert, BertOverNode2Vec
from graphmel.scripts.training.training import train_model
from graphmel.scripts.utils.io import load_node_id2terms_list, load_tuples, update_log_file, save_dict
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
    num_steps = 0
    for pos_batch_input_ids, pos_batch_attention_masks, neg_batch_input_ids, neg_batch_attention_masks in \
            tqdm(train_loader, miniters=len(train_loader) // 100):
        optimizer.zero_grad()
        loss = model.loss(pos_batch_input_ids.to(device), pos_batch_attention_masks.to(device),
                          neg_batch_input_ids.to(device), neg_batch_attention_masks.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_steps += 1
    return total_loss / len(train_loader), num_steps


def node2vec_val_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for pos_batch_input_ids, pos_batch_attention_masks, neg_batch_input_ids, neg_batch_attention_masks in \
                tqdm(val_loader, miniters=len(val_loader) // 100):
            loss = model.loss(pos_batch_input_ids.to(device), pos_batch_attention_masks.to(device),
                              neg_batch_input_ids.to(device), neg_batch_attention_masks.to(device))

            total_loss += loss.item()
    return total_loss / len(val_loader)


def main(args):
    output_dir = args.output_dir
    output_subdir = f"nns-c-wpn-p-q-wl_{args.node2vec_num_negative_samples}-{args.node2vec_walks_per_node}-" \
                    f"{args.node2vec_p}-{args.node2vec_q}-{args.node2vec_walk_length}_b{args.batch_size}_" \
                    f"lr{args.learning_rate}"
    output_dir = os.path.join(output_dir, output_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )

    train_node_id2terms_dict = load_node_id2terms_list(dict_path=args.train_node2terms_path, )
    train_edges_tuples = load_tuples(args.train_edges_path)
    val_node_id2terms_dict = load_node_id2terms_list(dict_path=args.val_node2terms_path, )
    val_edges_tuples = load_tuples(args.val_edges_path)

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    bert_encoder = AutoModel.from_pretrained(args.text_encoder)

    train_node_id2token_ids_dict = tokenize_node_terms(train_node_id2terms_dict, tokenizer,
                                                       max_length=args.text_encoder_seq_length)
    train_num_nodes = len(set(train_node_id2terms_dict.keys()))
    train_edge_index = convert_edges_tuples_to_edge_index(edges_tuples=train_edges_tuples)

    val_node_id2token_ids_dict = tokenize_node_terms(val_node_id2terms_dict, tokenizer,
                                                     max_length=args.text_encoder_seq_length)
    val_num_nodes = len(set(val_node_id2terms_dict.keys()))
    val_edge_index = convert_edges_tuples_to_edge_index(edges_tuples=val_edges_tuples)
    if args.debug:
        print("train_node_id2terms_dict:")
        for i, (k, v) in enumerate(train_node_id2terms_dict.items()):
            if i < 3:
                print(f"{k} ||| {v}")
        print(f"train_edges_tuples: {len(train_edges_tuples)}, {train_edges_tuples[:3]}")
        print("train_node_id2token_ids_dict:")
        for i, (k, v) in enumerate(train_node_id2token_ids_dict.items()):
            if i < 3:
                print(f"{k} ||| {v}")
        print(f"train_num_nodes: {train_num_nodes}")
        print(f"train_edge_index size: {train_edge_index.size()}")
        print('--' * 10)
        print("val_node_id2terms_dict:")
        for i, (k, v) in enumerate(val_node_id2terms_dict.items()):
            if i < 3:
                print(f"{k} ||| {v}")
        print(f"val_edges_tuples: {len(val_edges_tuples)}, {val_edges_tuples[:3]}")
        print("val_node_id2token_ids_dict:")
        for i, (k, v) in enumerate(val_node_id2token_ids_dict.items()):
            if i < 3:
                print(f"{k} ||| {v}")
        print(f"val_num_nodes: {val_num_nodes}")
        print(f"val_edge_index size: {val_edge_index.size()}")
    logging.info(f"There are {train_num_nodes} nodes in train and {val_num_nodes} nodes in validation")
    train_dataset = Node2vecDataset(edge_index=train_edge_index, node_id_to_token_ids_dict=train_node_id2token_ids_dict,
                                    walk_length=args.node2vec_walk_length, walks_per_node=args.node2vec_walks_per_node,
                                    p=args.node2vec_p, q=args.node2vec_q,
                                    num_negative_samples=args.node2vec_num_negative_samples,
                                    context_size=args.node2vec_context_size,
                                    num_nodes=train_num_nodes, seq_max_length=args.text_encoder_seq_length)
    val_dataset = Node2vecDataset(edge_index=val_edge_index, node_id_to_token_ids_dict=val_node_id2token_ids_dict,
                                  walk_length=10, walks_per_node=1,
                                  p=1, q=1,
                                  num_negative_samples=1,
                                  context_size=10,
                                  num_nodes=val_num_nodes, seq_max_length=args.text_encoder_seq_length)
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.sample, batch_size=args.batch_size,
                              num_workers=args.dataloader_num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn=val_dataset.sample, batch_size=args.batch_size,
                            num_workers=args.dataloader_num_workers, shuffle=False, drop_last=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multigpu_flag = False
    if args.gpus > 1:
        multigpu_flag = True

    model = BertOverNode2Vec(bert_encoder=bert_encoder, seq_max_length=args.text_encoder_seq_length,
                             multigpu_flag=multigpu_flag).to(device)

    train_model(model=model, train_epoch_fn=node2vec_train_epoch, val_epoch_fn=node2vec_val_epoch,
                chkpnt_path=args.model_checkpoint_path, train_loader=train_loader, val_loader=val_loader,
                learning_rate=args.learning_rate, num_epochs=args.num_epochs, output_dir=output_dir,
                save_chkpnt_epoch_interval=args.save_every_N_epoch, device=device)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--train_node2terms_path', type=str)
    parser.add_argument('--train_edges_path', type=str)
    parser.add_argument('--val_node2terms_path', type=str)
    parser.add_argument('--val_edges_path', type=str)
    # parser.add_argument('--train_proportion', type=float)
    parser.add_argument('--save_every_N_epoch', type=int, default=1)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--text_encoder_seq_length', type=int)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--node2vec_num_negative_samples', type=int, required=False)
    parser.add_argument('--node2vec_context_size', type=int, required=False)
    parser.add_argument('--node2vec_walks_per_node', type=int, required=False)
    parser.add_argument('--node2vec_p', type=float, required=False)
    parser.add_argument('--node2vec_q', type=float, required=False)
    parser.add_argument('--node2vec_walk_length', type=int)
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
