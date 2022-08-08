#!/usr/bin/env python
import gc
from itertools import product
from typing import Iterable, Tuple

import numpy as np
import argparse
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import logging
import time
import os
import random

from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from tqdm import tqdm

from graphmel.scripts.models.heterogeneous_graphsage_sapbert import HeteroGraphSAGESapMetricLearning
from graphmel.scripts.self_alignment_pretraining.dataset import PositivePairNeighborSampler, \
    PositiveRelationalNeighborSampler, HeterogeneousPositivePairNeighborSampler, graph_to_hetero_dataset
from graphmel.scripts.self_alignment_pretraining.graph_sapbert_models import RGCNDGISapMetricLearning
from graphmel.scripts.self_alignment_pretraining.sapbert_training import train_graph_sapbert_model
from graphmel.scripts.training.data.dataset import load_positive_pairs, map_terms2term_id, \
    create_term_id2tokenizer_output, load_data_and_bert_model, convert_edges_tuples_to_edge_index, \
    convert_edges_tuples_to_oriented_edge_index_with_relations
from graphmel.scripts.utils.io import save_dict, load_dict

# import wandb
# wandb.init(project="sapbert")
from graphmel.scripts.utils.umls2graph import get_unique_sem_group_edge_rel_combinations


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    # parser.add_argument('--model_dir',
    #                     help='Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='training set directory')
    # parser.add_argument('--val_dir', type=str, required=False,
    #                     help='Validation set directory')
    parser.add_argument('--validate', action="store_true",
                        help='whether the validation of each epoch is required')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')

    # Graphsage + DGI configuration
    parser.add_argument('--graphsage_num_neighbors', type=int, nargs='+')
    parser.add_argument('--num_graphsage_layers', type=int)
    parser.add_argument('--graphsage_hidden_channels', type=int)
    parser.add_argument('--graphsage_dropout_p', type=float, )
    parser.add_argument('--dgi_loss_weight', type=float)
    parser.add_argument('--remove_selfloops', action="store_true")

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--batch_size',
                        help='train batch size',
                        default=240, type=int)
    parser.add_argument('--num_epochs',
                        help='epoch to train',
                        default=3, type=int)
    parser.add_argument('--amp', action="store_true",
                        help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--random_seed',
                        help='',
                        default=1996, type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
                        default="ms_loss")
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument('--type_of_triplets', default="all", type=str)
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}")

    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--save_every_N_epoch', type=int, default=1)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)

    args = parser.parse_args()
    return args


def heterogeneous_graphsage_dgi_sapbert_train_step(model: HeteroGraphSAGESapMetricLearning, hetero_dataset: HeteroData,
                                                   batch, amp, device):
    term_1_input_ids, term_1_att_masks = batch["term_1_input"]
    term_1_input_ids, term_1_att_masks = term_1_input_ids.to(device), term_1_att_masks.to(device)
    term_2_input_ids, term_2_att_masks = batch["term_2_input"]
    term_2_input_ids, term_2_att_masks = term_2_input_ids.to(device), term_2_att_masks.to(device)
    edge_index = batch["edge_index"].to(device)
    src_semantic_groups = batch["src_semantic_groups"]
    trg_semantic_groups = batch["trg_semantic_groups"]
    batch_size = batch["batch_size"]
    concept_ids = batch["concept_ids"].to(device)
    rel_types = batch["rel_ids_list"]

    term_1_node_features = model.bert_encode(input_ids=term_1_input_ids, att_masks=term_1_att_masks)
    term_2_node_features = model.bert_encode(input_ids=term_2_input_ids, att_masks=term_2_att_masks)

    hetero_dataset = graph_to_hetero_dataset(edge_index=edge_index, hetero_dataset=hetero_dataset,
                                             node_features=term_1_node_features,
                                             all_node_types=hetero_dataset.all_node_types,
                                             sem_group_rel_combs=hetero_dataset.sem_group_rel_combs,
                                             src_node_sem_groups=src_semantic_groups,
                                             trg_node_sem_groups=trg_semantic_groups, rel_types=rel_types)

    dgi_loss_1 = model.dgi_loss(x_dict=hetero_dataset.x_dict, edge_index_dict=hetero_dataset.edge_index_dict,)
    hetero_dataset = graph_to_hetero_dataset(edge_index=edge_index, node_features=term_2_node_features,
                                             hetero_dataset=hetero_dataset,
                                             src_node_sem_groups=src_semantic_groups,
                                             all_node_types=hetero_dataset.all_node_types,
                                             sem_group_rel_combs=hetero_dataset.sem_group_rel_combs,
                                             trg_node_sem_groups=trg_semantic_groups, rel_types=rel_types)
    dgi_loss_2 = model.dgi_loss(hetero_dataset.x_dict, hetero_dataset.edge_index_dict, )

    if amp:
        with autocast():
            sapbert_loss = model(query_embed1=term_1_node_features, query_embed2=term_2_node_features,
                                 concept_ids=concept_ids, batch_size=batch_size)
    else:
        sapbert_loss = model(query_embed1=term_1_node_features, query_embed2=term_2_node_features,
                             concept_ids=concept_ids, batch_size=batch_size)
    # logging.info(f"sapbert_loss {sapbert_loss.item()}")
    # logging.info(f"dgi_loss_1 {dgi_loss_1.item()}")
    # logging.info(f"dgi_loss_2 {dgi_loss_2.item()}")

    loss = sapbert_loss + (dgi_loss_1 + dgi_loss_2) * model.dgi_loss_weight
    logging.info(f"loss {loss.item()}")
    return loss


def heterogeneous_graphsage_dgi_sapbert_eval_step(model: HeteroGraphSAGESapMetricLearning, batch, amp, device):
    term_1_input_ids, term_1_att_masks = batch["term_1_input"]
    term_1_input_ids, term_1_att_masks = term_1_input_ids.to(device), term_1_att_masks.to(device)
    term_2_input_ids, term_2_att_masks = batch["term_2_input"]
    term_2_input_ids, term_2_att_masks = term_2_input_ids.to(device), term_2_att_masks.to(device)

    concept_ids = batch["concept_ids"].to(device)
    batch_size = batch["batch_size"]

    if amp:
        with autocast():
            sapbert_loss = model.eval_step_loss(term_1_input_ids=term_1_input_ids, term_1_att_masks=term_1_att_masks,
                                                term_2_input_ids=term_2_input_ids, term_2_att_masks=term_2_att_masks,
                                                concept_ids=concept_ids, batch_size=batch_size)
    else:
        sapbert_loss = model.eval_step_loss(term_1_input_ids=term_1_input_ids, term_1_att_masks=term_1_att_masks,
                                            term_2_input_ids=term_2_input_ids, term_2_att_masks=term_2_att_masks,
                                            concept_ids=concept_ids, batch_size=batch_size)
    return sapbert_loss


def train_heterogeneous_graphsage_dgi_sapbert(model: HeteroGraphSAGESapMetricLearning,
                                              train_loader: HeterogeneousPositivePairNeighborSampler,
                                              optimizer: torch.optim.Optimizer, scaler, amp, device):
    model.train()
    total_loss = 0
    num_steps = 0
    for batch in tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader)):
        optimizer.zero_grad()
        loss = heterogeneous_graphsage_dgi_sapbert_train_step(model=model, hetero_dataset=train_loader.hetero_dataset,
                                                              batch=batch, amp=amp, device=device)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        num_steps += 1
        total_loss += float(loss)
        # wandb.log({"Train loss": loss.item()})
        # logging.info(f"num_steps  {num_steps}, Train step loss: {loss}")
    total_loss /= (num_steps + 1e-9)
    print("total_loss", total_loss)
    return total_loss, num_steps


def val_heterogeneous_graphsage_dgi_sapbert(model: HeteroGraphSAGESapMetricLearning,
                                            val_loader: HeterogeneousPositivePairNeighborSampler,
                                            amp, device):
    model.eval()
    total_loss = 0
    num_steps = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, miniters=len(val_loader) // 100, total=len(val_loader)):
            loss = heterogeneous_graphsage_dgi_sapbert_eval_step(model=model, batch=batch, amp=amp, device=device)
            num_steps += 1
            total_loss += float(loss)
            # wandb.log({"Val loss": loss.item()})
    total_loss /= (num_steps + 1e-9)
    return total_loss


def initialize_hetero_graph_sapbert_model(model: HeteroGraphSAGESapMetricLearning, all_node_types: Iterable[str],
                                          sem_group_rel_combs: Iterable[Tuple[str, str, int]],
                                          emb_size: int):
    logging.info("Initializing heterogeneous GraphSAGE model.")
    logging.info(f"Num possible node types: {len(all_node_types)}")
    logging.info(f"Num possible node, edge type combinations: {len(sem_group_rel_combs)}")

    het_data = HeteroData()
    for node_type in all_node_types:
        het_data[node_type].x = torch.zeros((1, emb_size), dtype=torch.float)
    for (node_type_1, node_type_2, rel_id) in sem_group_rel_combs:
        het_data[node_type_1, str(rel_id), node_type_2].edge_index = torch.zeros((2, 1), dtype=torch.long)
    model.hetero_graphsage = to_hetero(model.hetero_graphsage, het_data.metadata(), aggr="mean")
    model.initialize_graphsage_layers(het_data.x_dict, het_data.edge_index_dict)
    logging.info("Heterogeneous GraphSAGE model has been initialized.")
    return model, het_data


def main(args):
    print(args)
    output_dir = args.output_dir
    output_subdir = f"graphsage_n-{args.graphsage_num_neighbors}_l-{args.num_graphsage_layers}_" \
                    f"c-{args.graphsage_hidden_channels}_p-{args.graphsage_dropout_p}_dgi_{args.dgi_loss_weight}" \
                    f"lr_{args.learning_rate}_b_{args.batch_size}"
    output_dir = os.path.join(output_dir, output_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )
    torch.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.random.manual_seed(args.random_seed)
    torch.cuda.random.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    node2terms_path = os.path.join(args.train_dir, "node_id2terms_list")
    node_id2sem_group = os.path.join(args.train_dir, f"node_id2sem_group")
    edges_path = os.path.join(args.train_dir, "edges")

    bert_encoder, bert_tokenizer, node_id2token_ids_dict, edge_tuples, _, _ = \
        load_data_and_bert_model(train_node2terms_path=node2terms_path,
                                 train_edges_path=edges_path, use_fast=True, do_lower_case=True,
                                 val_node2terms_path=node2terms_path,
                                 val_edges_path=edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=args.max_length, drop_relations_info=False)

    del _

    edge_index, edge_rel_ids = \
        convert_edges_tuples_to_oriented_edge_index_with_relations(edge_tuples, use_rel_or_rela='rel',
                                                                   remove_selfloops=args.remove_selfloops)

    assert edge_index.size()[1] == len(edge_rel_ids)

    num_edges = edge_index.size()[1]
    num_nodes = len(set(node_id2token_ids_dict.keys()))
    node_id2sem_group = {int(node_id): sem_group for node_id, sem_group in load_dict(node_id2sem_group).items()}
    unique_sem_group_rel_combinations = get_unique_sem_group_edge_rel_combinations(node_id2sem_group, edge_tuples)
    del edge_tuples

    train_positive_pairs_path = os.path.join(args.train_dir, f"train_pos_pairs")
    train_pos_pairs_term_1_list, train_pos_pairs_term_2_list, train_pos_pairs_concept_ids = \
        load_positive_pairs(train_positive_pairs_path)
    train_pos_pairs_term_1_id_list, train_pos_pairs_term_2_id_list, train_term2id = map_terms2term_id(
        term_1_list=train_pos_pairs_term_1_list, term_2_list=train_pos_pairs_term_2_list)
    logging.info(f"There are {len(train_pos_pairs_term_1_id_list)} positive training pairs")
    train_term_id2tok_out = create_term_id2tokenizer_output(term2id=train_term2id, max_length=args.max_length,
                                                            tokenizer=bert_tokenizer)
    del train_pos_pairs_term_1_list
    del train_pos_pairs_term_2_list

    logging.info(f"There are {num_nodes} nodes and {num_edges} edges in graph.")
    train_num_pos_pairs = len(train_pos_pairs_term_1_id_list)
    train_pos_pairs_idx = torch.LongTensor(range(train_num_pos_pairs))
    train_pos_pair_sampler = HeterogeneousPositivePairNeighborSampler(
        pos_pairs_term_1_id_list=train_pos_pairs_term_1_id_list, edge_index=edge_index,
        pos_pairs_term_2_id_list=train_pos_pairs_term_2_id_list,
        pos_pairs_concept_ids_list=train_pos_pairs_concept_ids, sizes=args.graphsage_num_neighbors,
        node_id2sem_group=node_id2sem_group, term_id2tokenizer_output=train_term_id2tok_out, rel_ids=edge_rel_ids,
        node_idx=train_pos_pairs_idx, node_id2token_ids_dict=node_id2token_ids_dict, seq_max_length=args.max_length,
        batch_size=args.batch_size, num_workers=args.dataloader_num_workers, shuffle=True, )

    val_pos_pair_sampler = None
    val_epoch_fn = None
    if args.validate:
        val_positive_pairs_path = os.path.join(args.train_dir, f"val_pos_pairs")
        val_pos_pairs_term_1_list, val_pos_pairs_term_2_list, val_pos_pairs_concept_ids = \
            load_positive_pairs(val_positive_pairs_path)
        val_pos_pairs_term_1_id_list, val_pos_pairs_term_2_id_list, val_term2id = map_terms2term_id(
            term_1_list=val_pos_pairs_term_1_list, term_2_list=val_pos_pairs_term_2_list)
        logging.info(f"There are {len(val_pos_pairs_term_1_id_list)} positive validation pairs")
        del val_pos_pairs_term_1_list
        del val_pos_pairs_term_2_list
        val_term_id2tok_out = create_term_id2tokenizer_output(term2id=val_term2id, max_length=args.max_length,
                                                              tokenizer=bert_tokenizer)
        val_num_pos_pairs = len(val_pos_pairs_term_1_id_list)
        val_pos_pairs_idx = torch.LongTensor(range(val_num_pos_pairs))
        val_pos_pair_sampler = HeterogeneousPositivePairNeighborSampler(
            pos_pairs_term_1_id_list=val_pos_pairs_term_1_id_list, edge_index=edge_index,
            pos_pairs_term_2_id_list=val_pos_pairs_term_2_id_list,
            pos_pairs_concept_ids_list=val_pos_pairs_concept_ids, sizes=args.graphsage_num_neighbors,
            node_id2sem_group=node_id2sem_group, term_id2tokenizer_output=val_term_id2tok_out, rel_ids=edge_rel_ids,
            node_idx=val_pos_pairs_idx, node_id2token_ids_dict=node_id2token_ids_dict, seq_max_length=args.max_length,
            batch_size=args.batch_size, num_workers=args.dataloader_num_workers, shuffle=False, )
        val_epoch_fn = val_heterogeneous_graphsage_dgi_sapbert
    device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    model = HeteroGraphSAGESapMetricLearning(bert_encoder, num_graphsage_layers=args.num_graphsage_layers,
                                             graphsage_hidden_channels=args.graphsage_hidden_channels,
                                             graphsage_dropout_p=args.graphsage_dropout_p,
                                             dgi_loss_weight=args.dgi_loss_weight,
                                             use_cuda=args.use_cuda, loss=args.loss,
                                             multigpu_flag=args.parallel, use_miner=args.use_miner,
                                             miner_margin=args.miner_margin, type_of_triplets=args.type_of_triplets,
                                             agg_mode=args.agg_mode)

    all_node_types = list(set(node_id2sem_group.values()))
    # all_edge_types = list(set((t.item() for t in edge_rel_ids)))
    model, hetero_dataset = initialize_hetero_graph_sapbert_model(model=model, all_node_types=all_node_types,
                                                  sem_group_rel_combs=unique_sem_group_rel_combinations,
                                                  emb_size=bert_encoder.config.hidden_size)
    model = model.to(device)
    train_pos_pair_sampler.hetero_dataset = hetero_dataset
    hetero_dataset.all_node_types = all_node_types
    hetero_dataset.sem_group_rel_combs = unique_sem_group_rel_combinations
    start = time.time()
    train_graph_sapbert_model(model=model, train_epoch_fn=train_heterogeneous_graphsage_dgi_sapbert,
                              val_epoch_fn=val_epoch_fn, train_loader=train_pos_pair_sampler,
                              val_loader=val_pos_pair_sampler, learning_rate=args.learning_rate,
                              weight_decay=args.weight_decay, num_epochs=args.num_epochs, output_dir=output_dir,
                              save_chkpnt_epoch_interval=args.save_every_N_epoch,
                              amp=args.amp, scaler=scaler, device=device, chkpnt_path=args.model_checkpoint_path)
    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logging.info(f"Training Time took {training_hour} hours {training_minute} minutes {training_second} seconds")


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
