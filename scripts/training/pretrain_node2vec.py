import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphmel.scripts.training.data.dataset import Node2vecDataset, load_data_and_bert_model, convert_edges_tuples_to_edge_index
from graphmel.scripts.training.model import BertOverNode2Vec
from graphmel.scripts.training.training import train_model
from graphmel.scripts.utils.io import save_dict
import torch


# from torch_geometric.data import NeighborSampler as RawNeighborSampler


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
    output_subdir = f"nns-c-wpn-p-q-wl_{args.node2vec_train_num_negative_samples}-{args.node2vec_train_walks_per_node}-" \
                    f"{args.node2vec_train_p}-{args.node2vec_train_q}-{args.node2vec_train_walk_length}_b{args.batch_size}_" \
                    f"lr{args.learning_rate}"
    output_dir = os.path.join(output_dir, output_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )

    bert_encoder, train_node_id2token_ids_dict, train_edges_tuples, val_node_id2token_ids_dict, val_edges_tuples = \
        load_data_and_bert_model(train_node2terms_path=args.train_node2terms_path,
                                 train_edges_path=args.train_edges_path,
                                 val_node2terms_path=args.val_node2terms_path,
                                 val_edges_path=args.val_edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=args.text_encoder_seq_length, drop_relations_info=True)
    train_num_nodes = len(set(train_node_id2token_ids_dict.keys()))
    val_num_nodes = len(set(val_node_id2token_ids_dict.keys()))
    # train_edge_index = convert_edges_tuples_to_oriented_edge_index_with_relations(edges_tuples=train_edges_tuples)
    # val_edge_index = convert_edges_tuples_to_oriented_edge_index_with_relations(edges_tuples=val_edges_tuples)
    train_edge_index = convert_edges_tuples_to_edge_index(edges_tuples=train_edges_tuples)
    val_edge_index = convert_edges_tuples_to_edge_index(edges_tuples=val_edges_tuples)

    logging.info(f"There are {train_num_nodes} nodes in train and {val_num_nodes} nodes in validation")
    train_dataset = Node2vecDataset(edge_index=train_edge_index, node_id_to_token_ids_dict=train_node_id2token_ids_dict,
                                    walk_length=args.node2vec_train_walk_length,
                                    walks_per_node=args.node2vec_train_walks_per_node,
                                    p=args.node2vec_train_p, q=args.node2vec_train_q,
                                    num_negative_samples=args.node2vec_train_num_negative_samples,
                                    context_size=args.node2vec_train_context_size,
                                    num_nodes=train_num_nodes, seq_max_length=args.text_encoder_seq_length)

    val_dataset = Node2vecDataset(edge_index=val_edge_index, node_id_to_token_ids_dict=val_node_id2token_ids_dict,
                                  walk_length=args.node2vec_val_walk_length,
                                  walks_per_node=args.node2vec_val_walks_per_node,
                                  p=args.node2vec_val_p, q=args.node2vec_val_q,
                                  num_negative_samples=args.node2vec_val_num_negative_samples,
                                  context_size=args.node2vec_val_context_size,
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
    parser.add_argument('--node2vec_train_num_negative_samples', type=int, required=False)
    parser.add_argument('--node2vec_train_context_size', type=int, required=False)
    parser.add_argument('--node2vec_train_walks_per_node', type=int, required=False)
    parser.add_argument('--node2vec_train_p', type=float, required=False)
    parser.add_argument('--node2vec_train_q', type=float, required=False)
    parser.add_argument('--node2vec_train_walk_length', type=int)
    parser.add_argument('--node2vec_val_num_negative_samples', type=int, required=False, default=1)
    parser.add_argument('--node2vec_val_context_size', type=int, required=False, default=10)
    parser.add_argument('--node2vec_val_walks_per_node', type=int, required=False, default=1)
    parser.add_argument('--node2vec_val_p', type=float, required=False, default=1.)
    parser.add_argument('--node2vec_val_q', type=float, required=False, default=1.)
    parser.add_argument('--node2vec_val_walk_length', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_epochs', type=int)
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
