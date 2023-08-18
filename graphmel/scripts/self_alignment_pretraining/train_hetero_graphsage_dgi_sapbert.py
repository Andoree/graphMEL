#!/usr/bin/env python
import argparse
import logging
import os
import random
import time
from typing import Iterable, Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from tqdm import tqdm

from graphmel.scripts.models.heterogeneous_graphsage_dgi_sapbert import HeteroGraphSageDgiSapMetricLearning
from graphmel.scripts.self_alignment_pretraining.dataset import HeterogeneousPositivePairNeighborSampler, \
    HeterogeneousPositivePairNeighborSamplerV2
from graphmel.scripts.self_alignment_pretraining.sapbert_training import train_graph_sapbert_model
from graphmel.scripts.training.data.dataset import load_positive_pairs, map_terms2term_id, \
    create_term_id2tokenizer_output, load_data_and_bert_model, \
    convert_edges_tuples_to_oriented_edge_index_with_relations
from graphmel.scripts.utils.io import save_dict, load_dict
# import wandb
# wandb.init(project="sapbert")
from graphmel.scripts.utils.umls2graph import get_unique_sem_group_edge_rel_combinations, filter_edges


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert train')

    # Required
    # parser.add_argument('--model_dir',
    #                     help='Directory for pretrained model')
    # parser.add_argument('--train_dir', type=str, required=True,
    #                     help='training set directory')
    # # parser.add_argument('--val_dir', type=str, required=False,
    # #                     help='Validation set directory')
    # parser.add_argument('--validate', action="store_true",
    #                     help='whether the validation of each epoch is required')
    #
    # parser.add_argument('--output_dir', type=str, required=True,
    #                     help='Directory for output')

    # Graphsage + DGI configuration
    # parser.add_argument('--graphsage_num_neighbors', type=int, nargs='+')
    # parser.add_argument('--num_graphsage_layers', type=int)
    # parser.add_argument('--graphsage_hidden_channels', type=int)
    # parser.add_argument('--graphsage_dropout_p', type=float, )
    # parser.add_argument('--graph_loss_weight', type=float, )
    # parser.add_argument('--dgi_loss_weight', type=float)
    # # parser.add_argument('--filter_rel_types', action="store_true")
    #
    # parser.add_argument('--intermodal_loss_weight', type=float, required=False)
    # parser.add_argument('--use_intermodal_miner', action="store_true")
    # parser.add_argument('--modality_distance', type=str, required=False, choices=(None, "sapbert", "cosine", "MSE"))
    # parser.add_argument('--text_loss_weight', type=float, required=False, default=1.0)
    # parser.add_argument('--freeze_non_target_nodes', action="store_true")
    #
    # # Tokenizer settings
    # parser.add_argument('--max_length', default=25, type=int)
    #
    # # Train config
    # parser.add_argument('--use_cuda', action="store_true")
    # parser.add_argument('--learning_rate',
    #                     help='learning rate',
    #                     default=0.0001, type=float)
    # parser.add_argument('--weight_decay',
    #                     help='weight decay',
    #                     default=0.01, type=float)
    # parser.add_argument('--batch_size',
    #                     help='train batch size',
    #                     default=240, type=int)
    # parser.add_argument('--num_epochs',
    #                     help='epoch to train',
    #                     default=3, type=int)
    # parser.add_argument('--amp', action="store_true",
    #                     help="automatic mixed precision training")
    # parser.add_argument('--parallel', action="store_true")
    # parser.add_argument('--random_seed',
    #                     help='',
    #                     default=1996, type=int)
    # parser.add_argument('--loss',
    #                     help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
    #                     default="ms_loss")
    # parser.add_argument('--use_miner', action="store_true")
    # parser.add_argument('--miner_margin', default=0.2, type=float)
    # parser.add_argument('--type_of_triplets', default="all", type=str)
    # parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}")
    #
    # parser.add_argument('--text_encoder', type=str)
    # parser.add_argument('--dataloader_num_workers', type=int)
    # parser.add_argument('--save_every_N_epoch', type=int, default=1)
    # parser.add_argument('--model_checkpoint_path', required=False, default=None)

    parser.add_argument('--train_dir', type=str, required=True,
                        help='training set directory',
                        default="/home/c204/University/NLP/SCR_data/SCR_SCR_FULL")
    # parser.add_argument('--val_dir', type=str, required=False,
    #                     help='Validation set directory')
    parser.add_argument('--validate', action="store_true",
                        help='whether the validation of each epoch is required')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output', default="DELETE/")

    parser.add_argument('--graphsage_num_neighbors', type=int, nargs='+', default=1)
    parser.add_argument('--num_graphsage_layers', type=int, default=3)
    parser.add_argument('--graphsage_hidden_channels', type=int, default=324)
    parser.add_argument('--graphsage_dropout_p', type=float, default=0.1)
    parser.add_argument('--graph_loss_weight', type=float, default=1.0)
    parser.add_argument('--dgi_loss_weight', type=float, default=1.0)
    # parser.add_argument('--filter_rel_types', action="store_true")

    parser.add_argument('--intermodal_loss_weight', type=float, required=False, default=1.0)
    parser.add_argument('--use_intermodal_miner', action="store_true", default=True)
    parser.add_argument('--modality_distance', type=str, required=False, default="sapbert",
                        choices=(None, "sapbert", "cosine", "MSE"))
    parser.add_argument('--text_loss_weight', type=float, required=False, default=1.0)
    parser.add_argument('--freeze_non_target_nodes', action="store_true")

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
                        default=2, type=int)
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

    parser.add_argument('--text_encoder', type=str, default="prajjwal1/bert-tiny")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--save_every_N_epoch', type=int, default=1)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)

    args = parser.parse_args()
    return args


def heterogeneous_graphsage_dgi_sapbert_train_step(model: HeteroGraphSageDgiSapMetricLearning, batch, amp, device,
                                                   freeze_non_target_nodes):
    term_1_input_ids, term_1_att_masks = batch["term_1_input"]
    term_1_input_ids, term_1_att_masks = term_1_input_ids.to(device), term_1_att_masks.to(device)
    term_2_input_ids, term_2_att_masks = batch["term_2_input"]
    term_2_input_ids, term_2_att_masks = term_2_input_ids.to(device), term_2_att_masks.to(device)
    concept_ids = batch["concept_ids"].to(device)
    nodes_bert_input = batch["nodes_bert_input"]
    batch_size = batch["batch_size"]
    hetero_dataset = batch["hetero_dataset"]

    for node_type, bert_input in nodes_bert_input.items():
        bert_input_ids, bert_att_masks = bert_input
        if freeze_non_target_nodes:
            with torch.no_grad():
                model.eval()
                bert_input_ids, bert_att_masks = bert_input_ids.to(device), bert_att_masks.to(device)
                bert_features = model.bert_encode(input_ids=bert_input_ids, att_masks=bert_att_masks).detach()
                model.train()
        else:
            bert_input_ids, bert_att_masks = bert_input_ids.to(device), bert_att_masks.to(device)
            bert_features = model.bert_encode(input_ids=bert_input_ids, att_masks=bert_att_masks)


        if node_type != "SRC":
            hetero_dataset[node_type].x = bert_features

    term_1_node_features = model.bert_encode(input_ids=term_1_input_ids, att_masks=term_1_att_masks)
    term_2_node_features = model.bert_encode(input_ids=term_2_input_ids, att_masks=term_2_att_masks)
    hetero_dataset["SRC"].x = torch.cat([term_1_node_features, term_2_node_features], dim=0)

    hetero_dataset = hetero_dataset.to(device)
    for edge_type, edge_index in hetero_dataset.edge_index_dict.items():
        if edge_type[-1] == "SRC":

            edge_index_copy = edge_index.clone()
            edge_index_copy[1, :] = batch_size + edge_index_copy[1, :]
            new_edge_index = torch.cat((edge_index, edge_index_copy), dim=1)
            hetero_dataset[edge_type].edge_index = new_edge_index


    pos_graph_embed, neg_graph_embed, graph_summary = model.graph_encode(x_dict=hetero_dataset.x_dict,
                                                                               edge_index_dict=hetero_dataset.edge_index_dict,
                                                                               batch_size=batch_size * 2, )
    pos_graph_embed_1 = pos_graph_embed[:batch_size, :]
    pos_graph_embed_2 = pos_graph_embed[batch_size:, :]
    neg_graph_embed_1 = neg_graph_embed[:batch_size, :]
    neg_graph_embed_2 = neg_graph_embed[batch_size:, :]
    graph_summary_1 = graph_summary
    graph_summary_2 = graph_summary

    # hetero_dataset["SRC"].x = term_2_node_features
    #
    #
    if amp:
        with autocast():
            sapbert_loss, graph_loss, dgi_loss, intermodal_loss = model(text_embed_1=term_1_node_features,
                                                                        text_embed_2=term_2_node_features,
                                                                        concept_ids=concept_ids,
                                                                        pos_graph_embed_1=pos_graph_embed_1,
                                                                        pos_graph_embed_2=pos_graph_embed_2,
                                                                        neg_graph_embed_1=neg_graph_embed_1,
                                                                        neg_graph_embed_2=neg_graph_embed_2,
                                                                        graph_summary_1=graph_summary_1,
                                                                        graph_summary_2=graph_summary_2,
                                                                        batch_size=batch_size)
    else:
        sapbert_loss, graph_loss, dgi_loss, intermodal_loss = model(text_embed_1=term_1_node_features,
                                                                        text_embed_2=term_2_node_features,
                                                                        concept_ids=concept_ids,
                                                                        pos_graph_embed_1=pos_graph_embed_1,
                                                                        pos_graph_embed_2=pos_graph_embed_2,
                                                                        neg_graph_embed_1=neg_graph_embed_1,
                                                                        neg_graph_embed_2=neg_graph_embed_2,
                                                                        graph_summary_1=graph_summary_1,
                                                                        graph_summary_2=graph_summary_2,
                                                                        batch_size=batch_size)

    return sapbert_loss, graph_loss, dgi_loss, intermodal_loss


def train_heterogeneous_graphsage_dgi_sapbert(model: HeteroGraphSageDgiSapMetricLearning,
                                              train_loader: HeterogeneousPositivePairNeighborSampler,
                                              optimizer: torch.optim.Optimizer, scaler, amp, device, **kwargs):

    model.train()
    freeze_non_target_nodes = kwargs["freeze_non_target_nodes"]
    # total_loss = 0
    losses_dict = {"total": 0, "sapbert": 0, "graph": 0, "dgi": 0, "intermodal": 0}
    num_steps = 0
    pbar = tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader))
    for batch in pbar:
        optimizer.zero_grad()
        sapbert_loss, graph_loss, dgi_loss, intermodal_loss = \
            heterogeneous_graphsage_dgi_sapbert_train_step(model=model, batch=batch, amp=amp, device=device,
                                                           freeze_non_target_nodes=freeze_non_target_nodes)
        sapbert_loss = sapbert_loss * model.sapbert_loss_weight
        graph_loss = graph_loss * model.graph_loss_weight
        dgi_loss = dgi_loss * model.dgi_loss_weight
        intermodal_loss = intermodal_loss * model.intermodal_loss_weight
        if intermodal_loss is not None:
            intermodal_loss = intermodal_loss * model.intermodal_loss_weight
            loss = sapbert_loss + graph_loss + dgi_loss + intermodal_loss
        else:
            loss = sapbert_loss + graph_loss + dgi_loss
            intermodal_loss = -1.
        pbar.set_description(f"L: {float(loss):.5f} ({float(sapbert_loss):.5f} + "
                             f"{float(graph_loss):.5f} + {float(dgi_loss):.5f} + {float(intermodal_loss):.5f})")
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        num_steps += 1
        losses_dict["total"] += float(loss)
        losses_dict["sapbert"] += float(sapbert_loss)
        losses_dict["graph"] += float(graph_loss)
        losses_dict["dgi"] += float(dgi_loss)
        losses_dict["intermodal"] += float(intermodal_loss)
    # total_loss /= (num_steps + 1e-9)
    losses_dict = {key: lo / (num_steps + 1e-9) for key, lo in losses_dict.items()}

    return losses_dict["total"], num_steps


def initialize_hetero_graph_sapbert_model(model: HeteroGraphSageDgiSapMetricLearning, hetero_dataset: HeteroData,
                                          emb_size: int):
    logging.info("Initializing heterogeneous GraphSAGE model.")
    all_node_types = hetero_dataset.node_types
    all_edge_types = hetero_dataset.edge_types
    logging.info(f"Num possible node types: {len(all_node_types)}")
    logging.info(f"Num possible node, edge type combinations: {len(all_edge_types)}")

    het_data = HeteroData()
    for node_type in all_node_types:
        het_data[node_type].x = torch.zeros((1, emb_size), dtype=torch.float)
    for node_type in all_node_types:
        het_data[node_type, "SELF-LOOP", node_type].edge_index = torch.zeros((2, 1), dtype=torch.long)
    for (node_type_1, rel_id, node_type_2) in all_edge_types:
        het_data[node_type_1, str(rel_id), node_type_2].edge_index = torch.zeros((2, 1), dtype=torch.long)
    # het_data = T.AddSelfLoops()(het_data)
    # for key, val in het_data.edge_index_dict.items():
    #     logging.info(f"initialize_hetero_graph_sapbert_model {key}={val}")
    model.hetero_graphsage = to_hetero(model.hetero_graphsage, het_data.metadata(), aggr="mean")
    model.eval()

    with torch.no_grad():
        model.hetero_graphsage(het_data.x_dict, het_data.edge_index_dict)
    logging.info("Heterogeneous GraphSAGE model has been initialized.")


# def initialize_hetero_graph_sapbert_model(model: HeteroGraphSageDgiSapMetricLearning, hetero_dataset: HeteroData,
#                                           emb_size: int):
#     logging.info("Initializing heterogeneous GraphSAGE model.")
#     all_node_types = hetero_dataset.node_types
#     all_edge_types = hetero_dataset.edge_types
#     logging.info(f"Num possible node types: {len(all_node_types)}")
#     logging.info(f"Num possible node, edge type combinations: {len(all_edge_types)}")
#
#     het_data = HeteroData()
#     for node_type in all_node_types:
#         het_data[node_type].x = torch.zeros((1, emb_size), dtype=torch.float)
#     for (node_type_1, rel_id, node_type_2) in all_edge_types:
#         het_data[node_type_1, str(rel_id), node_type_2].edge_index = torch.zeros((2, 1), dtype=torch.long)
#
#     model.hetero_graphsage = to_hetero(model.hetero_graphsage, het_data.metadata(), aggr="mean")
#     model.eval()
#     with torch.no_grad():
#         model.hetero_graphsage(het_data.x_dict, het_data.edge_index_dict)
#     logging.info("Heterogeneous GraphSAGE model has been initialized.")


def main(args):
    print(args)
    output_dir = args.output_dir

    output_subdir = f"graphsage_n-{args.graphsage_num_neighbors}_l-{args.num_graphsage_layers}_" \
                    f"c-{args.graphsage_hidden_channels}_p-{args.graphsage_dropout_p}" \
                    f"_text_{args.text_loss_weight}_graph_{args.graph_loss_weight}_intermodal_{args.modality_distance}" \
                    f"_{args.intermodal_loss_weight}_dgi_{args.dgi_loss_weight}" \
                    f"_intermodal_miner_{args.use_intermodal_miner}_freeze_neigh_{args.freeze_non_target_nodes}" \
                    f"_lr_{args.learning_rate}_b_{args.batch_size}"
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
    rel2id_path = os.path.join(args.train_dir, f"rel2id")
    edges_path = os.path.join(args.train_dir, "edges")
    rel2id = {rel: int(i) for rel, i in load_dict(rel2id_path).items()}

    bert_encoder, bert_tokenizer, node_id2token_ids_dict, edge_tuples, _, _ = \
        load_data_and_bert_model(train_node2terms_path=node2terms_path,
                                 train_edges_path=edges_path, use_fast=True, do_lower_case=True,
                                 val_node2terms_path=node2terms_path,
                                 val_edges_path=edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=args.max_length, drop_relations_info=False,
                                 tokenization_type="faster")

    del _
    # if args.filter_rel_types:
    #     edge_tuples = filter_edges(edge_tuples=edge_tuples, rel2id=rel2id)
    edge_index, edge_rel_ids = \
        convert_edges_tuples_to_oriented_edge_index_with_relations(edge_tuples, use_rel_or_rela='rel',
                                                                   remove_selfloops=True)

    assert edge_index.size()[1] == len(edge_rel_ids)

    num_edges = edge_index.size()[1]

    node_id2sem_group = {int(node_id): sem_group for node_id, sem_group in load_dict(node_id2sem_group).items()}
    node_id2token_ids_dict = {node_id: token_ids for node_id, token_ids in node_id2token_ids_dict.items()
                              if node_id2sem_group.get(node_id) is not None}
    num_nodes = len(set(node_id2token_ids_dict.keys()))
    unique_sem_group_rel_combinations = get_unique_sem_group_edge_rel_combinations(node_id2sem_group, edge_tuples)
    del edge_tuples

    train_positive_pairs_path = os.path.join(args.train_dir, f"train_pos_pairs")
    train_pos_pairs_term_1_list, train_pos_pairs_term_2_list, train_pos_pairs_concept_ids = \
        load_positive_pairs(train_positive_pairs_path)
    train_pos_pairs_term_1_id_list, train_pos_pairs_term_2_id_list, train_term2id = map_terms2term_id(
        term_1_list=train_pos_pairs_term_1_list, term_2_list=train_pos_pairs_term_2_list)
    logging.info(f"There are {len(train_pos_pairs_term_1_id_list)} positive training pairs")
    train_term_id2tok_out = create_term_id2tokenizer_output(term2id=train_term2id, max_length=args.max_length,
                                                            tokenizer=bert_tokenizer, code_version="faster")
    del train_pos_pairs_term_1_list
    del train_pos_pairs_term_2_list

    logging.info(f"There are {num_nodes} nodes and {num_edges} edges in graph.")
    train_num_pos_pairs = len(train_pos_pairs_term_1_id_list)
    logging.info(f"There are {train_num_pos_pairs} positive pairs in training set")

    train_pos_pair_sampler = HeterogeneousPositivePairNeighborSamplerV2(
        pos_pairs_term_1_id_list=train_pos_pairs_term_1_id_list, edge_index=edge_index,
        pos_pairs_term_2_id_list=train_pos_pairs_term_2_id_list, num_samples=args.graphsage_num_neighbors,
        pos_pairs_concept_ids_list=train_pos_pairs_concept_ids,
        node_id2sem_group=node_id2sem_group, term_id2tokenizer_output=train_term_id2tok_out, rel_ids=edge_rel_ids,
        node_id2token_ids_dict=node_id2token_ids_dict, seq_max_length=args.max_length,
        batch_size=args.batch_size, num_workers=args.dataloader_num_workers, shuffle=True,
        emb_size=bert_encoder.config.hidden_size)

    val_pos_pair_sampler = None
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
        # val_pos_pairs_idx = torch.LongTensor(range(val_num_pos_pairs))
        val_pos_pair_sampler = HeterogeneousPositivePairNeighborSamplerV2(
            pos_pairs_term_1_id_list=val_pos_pairs_term_1_id_list, edge_index=edge_index,
            pos_pairs_term_2_id_list=val_pos_pairs_term_2_id_list,
            pos_pairs_concept_ids_list=val_pos_pairs_concept_ids, num_samples=args.graphsage_num_neighbors,
            node_id2sem_group=node_id2sem_group, term_id2tokenizer_output=val_term_id2tok_out, rel_ids=edge_rel_ids,
            node_id2token_ids_dict=node_id2token_ids_dict, seq_max_length=args.max_length,
            batch_size=args.batch_size, num_workers=args.dataloader_num_workers, shuffle=False,
            emb_size=bert_encoder.config.hidden_size)

    device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    model = HeteroGraphSageDgiSapMetricLearning(bert_encoder, num_graphsage_layers=args.num_graphsage_layers,
                                                graphsage_hidden_channels=args.graphsage_hidden_channels,
                                                graphsage_dropout_p=args.graphsage_dropout_p,
                                                sapbert_loss_weight=args.text_loss_weight,
                                                graph_loss_weight=args.graph_loss_weight,
                                                dgi_loss_weight=args.dgi_loss_weight,
                                                modality_distance=args.modality_distance,
                                                intermodal_loss_weight=args.intermodal_loss_weight,
                                                use_cuda=args.use_cuda, loss=args.loss,
                                                multigpu_flag=args.parallel, use_miner=args.use_miner,
                                                miner_margin=args.miner_margin, type_of_triplets=args.type_of_triplets,
                                                agg_mode=args.agg_mode, use_intermodal_miner=args.use_intermodal_miner)

    initialize_hetero_graph_sapbert_model(model, hetero_dataset=train_pos_pair_sampler.hetero_dataset,
                                          emb_size=bert_encoder.config.hidden_size)

    model = model.to(device)

    start = time.time()
    train_graph_sapbert_model(model=model, train_epoch_fn=train_heterogeneous_graphsage_dgi_sapbert,
                              train_loader=train_pos_pair_sampler,
                              val_loader=val_pos_pair_sampler, learning_rate=args.learning_rate,
                              weight_decay=args.weight_decay, num_epochs=args.num_epochs, output_dir=output_dir,
                              save_chkpnt_epoch_interval=args.save_every_N_epoch,
                              amp=args.amp, scaler=scaler, device=device, chkpnt_path=args.model_checkpoint_path,
                              parallel=args.parallel, freeze_non_target_nodes=args.freeze_non_target_nodes)
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
