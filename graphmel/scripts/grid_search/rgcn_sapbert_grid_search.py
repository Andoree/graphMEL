#!/usr/bin/env python
import codecs
import itertools

import numpy as np
import argparse
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import logging
import time
import os
import random
from tqdm import tqdm
from transformers import AutoModel

from graphmel.scripts.evaluation.evaluate_all_checkpoints_in_dir import evaluate_single_checkpoint_acc1_acc5
from graphmel.scripts.evaluation.utils import read_dataset, read_vocab
from graphmel.scripts.self_alignment_pretraining.dataset import PositivePairNeighborSampler, \
    PositiveRelationalNeighborSampler
from graphmel.scripts.self_alignment_pretraining.graph_sapbert_models import GraphSAGESapMetricLearning, \
    RGCNSapMetricLearning
from graphmel.scripts.self_alignment_pretraining.sapbert_training import train_graph_sapbert_model
from graphmel.scripts.self_alignment_pretraining.train_rgcn_sapbert import train_rgcn_sapbert
from graphmel.scripts.training.data.data_utils import create_rel_id2inverse_rel_id_map
from graphmel.scripts.training.data.dataset import load_positive_pairs, map_terms2term_id, \
    create_term_id2tokenizer_output, load_data_and_bert_model, convert_edges_tuples_to_edge_index, \
    convert_edges_tuples_to_oriented_edge_index_with_relations
from graphmel.scripts.utils.io import save_dict, load_dict, save_encoder_from_checkpoint


# import wandb
# wandb.init(project="sapbert")


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


    # RGCN configuration
    parser.add_argument('--rgcn_num_hidden_channels', type=int, nargs='+')
    parser.add_argument('--rgcn_num_outer_layers', type=int, nargs='+')
    parser.add_argument('--rgcn_num_inner_layers', type=int, nargs='+')
    parser.add_argument('--rgcn_num_blocks', type=int, nargs='+')
    parser.add_argument('--rgcn_use_fast_conv', action="store_true")
    parser.add_argument('--rgcn_dropout_p', type=float, nargs='+')
    parser.add_argument('--rgcn_num_neighbors', type=int, nargs='+')
    parser.add_argument('--graph_loss_weight', type=float, nargs='+', )
    parser.add_argument('--remove_selfloops', action="store_true")
    parser.add_argument('--text_loss_weight', type=float, nargs='+', required=False, default=(1.0,))
    parser.add_argument('--intermodal_loss_weight', type=float, nargs='+', )
    parser.add_argument('--modality_distance', type=str, nargs='+', )
    parser.add_argument('--batch_size', type=int, nargs='+')

    parser.add_argument('--train_subset_ratio', type=float, )
    # Evaluation data path
    parser.add_argument('--data_folder', help='Path to the directory containing BioSyn format dataset', type=str,
                        nargs='+')
    parser.add_argument('--vocab', help='Path to the vocabulary file in BioSyn format', type=str, nargs='+')
    parser.add_argument('--eval_dataset_name', type=str, nargs='+')

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
    # parser.add_argument('--batch_size',
    #                     help='train batch size',
    #                     default=240, type=int)
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



def main(args):

    # TODO:
    """
    parser.add_argument('--rgcn_use_fast_conv', action="store_true")

    parser.add_argument('--remove_selfloops', action="store_true")
    parser.add_argument('--text_loss_weight', type=float, nargs='+', required=False, default=(1.0,))
    parser.add_argument('--intermodal_loss_weight', type=float, nargs='+', )
    parser.add_argument('--modality_distance', type=str, nargs='+', )
    parser.add_argument('--batch_size', type=int, nargs='+')

    """
    param_grid = {
        "rgcn_num_hidden_channels": args.rgcn_num_hidden_channels,
        "rgcn_num_outer_layers": args.rgcn_num_outer_layers,
        "rgcn_num_inner_layers": args.rgcn_num_inner_layers,
        "rgcn_num_blocks": args.rgcn_num_blocks,
        "rgcn_dropout_p": args.rgcn_dropout_p,
        "rgcn_num_neighbors": args.rgcn_num_neighbors,
        "graph_loss_weight": args.graph_loss_weight,
        "text_loss_weight": args.text_loss_weight,
        "intermodal_loss_weight": args.intermodal_loss_weight,
        "modality_distance": args.modality_distance,
        "batch_size": args.batch_size,
    }
    print(args)
    node2terms_path = os.path.join(args.train_dir, "node_id2terms_list")
    edges_path = os.path.join(args.train_dir, "edges")
    rel2id_path = os.path.join(args.train_dir, "rel2id")



    bert_encoder, bert_tokenizer, node_id2token_ids_dict, edges_tuples, _, _ = \
        load_data_and_bert_model(train_node2terms_path=node2terms_path,
                                 train_edges_path=edges_path, use_fast=True, do_lower_case=True,
                                 val_node2terms_path=node2terms_path,
                                 val_edges_path=edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=args.max_length, drop_relations_info=False)

    del _

    rel2id = load_dict(rel2id_path)

    num_relations = len(rel2id.keys())


    edge_index, edge_rel_ids = \
        convert_edges_tuples_to_oriented_edge_index_with_relations(edges_tuples, "rel",
                                                                   remove_selfloops=args.remove_selfloops )
    assert edge_index.size()[1] == len(edge_rel_ids)

    num_edges = edge_index.size()[1]
    num_nodes = len(set(node_id2token_ids_dict.keys()))

    del edges_tuples

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

    train_pos_pairs_term_1_id_list = np.array(train_pos_pairs_term_1_id_list)
    train_pos_pairs_term_2_id_list = np.array(train_pos_pairs_term_2_id_list)
    train_pos_pairs_concept_ids = np.array(train_pos_pairs_concept_ids)
    assert len(train_pos_pairs_term_1_id_list) == len(train_pos_pairs_term_2_id_list) \
           == len(train_pos_pairs_concept_ids)
    overall_num_pos_pairs = len(train_pos_pairs_term_1_id_list)
    perm = np.random.permutation(overall_num_pos_pairs)

    train_subset_ratio = args.train_subset_ratio
    train_subset_size = int(overall_num_pos_pairs * train_subset_ratio)
    selected_pos_pair_ids = perm[:train_subset_size]

    train_pos_pairs_term_1_id_list = train_pos_pairs_term_1_id_list[selected_pos_pair_ids]
    train_pos_pairs_term_2_id_list = train_pos_pairs_term_2_id_list[selected_pos_pair_ids]
    train_pos_pairs_concept_ids = train_pos_pairs_concept_ids[selected_pos_pair_ids]

    logging.info(f"There are {num_nodes} nodes and {num_edges} edges in graph.")
    train_num_pos_pairs = len(train_pos_pairs_term_1_id_list)
    train_pos_pairs_idx = torch.LongTensor(range(train_num_pos_pairs))
    logging.info(f"There are {train_num_pos_pairs} positive pairs")

    val_pos_pair_sampler = None
    val_epoch_fn = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    best_accs_dict = {}
    for dataset_name in args.eval_dataset_name:
        best_accs_dict[dataset_name] = {
            "acc_1": -1.,
            "acc_5": -1.,
        }

    best_params_dict = {}
    for dataset_name in args.eval_dataset_name:
        best_params_dict[dataset_name] = {
            "acc_1": {},
            "acc_5": {},
        }

    param_names = sorted(param_grid.keys())
    param_values_list = [param_grid[p_name] for p_name in param_names]

    for model_setup in itertools.product(*param_values_list):
        param_dict = {name: val for name, val in zip(param_names, model_setup)}

        rgcn_num_hidden_channels = param_dict["rgcn_num_hidden_channels"]
        rgcn_num_outer_layers = param_dict["rgcn_num_outer_layers"]
        rgcn_num_inner_layers = param_dict["rgcn_num_inner_layers"]

        rgcn_num_blocks = param_dict["rgcn_num_blocks"]
        rgcn_dropout_p = param_dict["rgcn_dropout_p"]
        rgcn_num_neighbors = (param_dict["rgcn_num_neighbors"],) * rgcn_num_outer_layers
        text_loss_weight = param_dict["text_loss_weight"]
        graph_loss_weight = param_dict["graph_loss_weight"]
        intermodal_loss_weight = param_dict["intermodal_loss_weight"]
        modality_distance = param_dict["modality_distance"]
        if modality_distance == "None":
            modality_distance = None

        batch_size = param_dict["batch_size"]
        logging.info("Processing configuration:")
        for k, v in param_dict.items():
            logging.info(f"{k}={v}")

        base_dir = args.output_dir
        conv_type = "fast_rgcn_conv" if args.rgcn_use_fast_conv else "rgcn_conv"

        output_subdir = f"rgcn_{rgcn_num_neighbors}_{rgcn_num_hidden_channels}-{rgcn_num_outer_layers}" \
                        f"-{rgcn_num_inner_layers}_{rgcn_num_blocks}_tl_{text_loss_weight}_gl_{graph_loss_weight}" \
                        f"intermodal_{modality_distance}_{intermodal_loss_weight}_rel_lr_{args.learning_rate}" \
                        f"_{conv_type}_{args.remove_selfloops}_drop_{rgcn_dropout_p}_b_{batch_size}"
        output_dir = os.path.join(base_dir, output_subdir)
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

        train_pos_pair_sampler = PositiveRelationalNeighborSampler(pos_pairs_term_1_id_list=train_pos_pairs_term_1_id_list,
                                                                   pos_pairs_term_2_id_list=train_pos_pairs_term_2_id_list,
                                                                   pos_pairs_concept_ids_list=train_pos_pairs_concept_ids,
                                                                   sizes=rgcn_num_neighbors, edge_index=edge_index,
                                                                   term_id2tokenizer_output=train_term_id2tok_out,
                                                                   rel_ids=edge_rel_ids, node_idx=train_pos_pairs_idx,
                                                                   node_id2token_ids_dict=node_id2token_ids_dict,
                                                                   seq_max_length=args.max_length,
                                                                   batch_size=batch_size,
                                                                   num_workers=args.dataloader_num_workers, shuffle=True, )


        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None
        bert_encoder = AutoModel.from_pretrained(args.text_encoder, )

        model = RGCNSapMetricLearning(bert_encoder=bert_encoder, rgcn_num_hidden_channels=rgcn_num_hidden_channels,
                                      rgcn_num_outer_layers=rgcn_num_outer_layers, rgcn_num_inner_layers=rgcn_num_inner_layers,
                                      rgcn_dropout_p=rgcn_dropout_p, sapbert_loss_weight=text_loss_weight,
                                      graph_loss_weight=graph_loss_weight, modality_distance=modality_distance,
                                      intermodal_loss_weight=intermodal_loss_weight,
                                      num_relations=num_relations, num_bases=None,
                                      num_blocks=rgcn_num_blocks, use_fast_conv=args.rgcn_use_fast_conv,
                                      use_cuda=args.use_cuda, loss=args.loss, miner_margin=args.miner_margin,
                                      type_of_triplets=args.type_of_triplets, agg_mode=args.agg_mode,
                                      multigpu_flag=args.parallel, ).to(device)
        start = time.time()
        try:
            train_graph_sapbert_model(model=model, train_epoch_fn=train_rgcn_sapbert,
                                      train_loader=train_pos_pair_sampler,
                                      val_loader=val_pos_pair_sampler,
                                      learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                      num_epochs=args.num_epochs, output_dir=output_dir, save_chkpnts=False,
                                      save_chkpnt_epoch_interval=args.save_every_N_epoch, parallel=args.parallel,
                                      amp=args.amp, scaler=scaler, device=device, chkpnt_path=args.model_checkpoint_path)
        except Exception:
            model = model.cpu()
            del model
            torch.cuda.empty_cache()
            continue
        end = time.time()
        training_time = end - start
        training_hour = int(training_time / 60 / 60)
        training_minute = int(training_time / 60 % 60)
        training_second = int(training_time % 60)
        logging.info(f"Training Time took {training_hour} hours {training_minute} minutes {training_second} seconds")

        checkpoint_path = os.path.join(output_dir, "final_checkpoint/")
        bert = AutoModel.from_pretrained(args.text_encoder, )
        if args.parallel:
            bert.load_state_dict(model.bert_encoder.module.state_dict())
        else:
            bert.load_state_dict(model.bert_encoder.state_dict())
        save_encoder_from_checkpoint(bert_encoder=bert, bert_tokenizer=bert_tokenizer,
                                     save_path=checkpoint_path)
        model = model.cpu()
        del model
        logging.info(f"Processing checkpoint: {checkpoint_path}")

        for data_folder, vocab, dataset_name in zip(args.data_folder, args.vocab, args.eval_dataset_name):
            entities = read_dataset(data_folder)
            ################
            entity_texts = [e['entity_text'].lower() for e in entities]
            labels = [e['label'] for e in entities]
            ##################
            vocab = read_vocab(vocab)

            acc_1, acc_5 = evaluate_single_checkpoint_acc1_acc5(checkpoint_path=checkpoint_path, vocab=vocab,
                                                                entity_texts=entity_texts, labels=labels)

            s = ','.join((f"{k}={v}" for k, v in param_dict.items()))
            s = f"{acc_1},{acc_5}\t{s}"
            grid_search_log_path = os.path.join(args.output_dir, f'grid_search_{dataset_name}.log')
            if acc_1 > best_accs_dict[dataset_name]["acc_1"]:
                best_accs_dict[dataset_name]["acc_1"] = acc_1
                best_params_dict[dataset_name]["acc_1"] = param_dict
            if acc_5 > best_accs_dict[dataset_name]["acc_5"]:
                best_accs_dict[dataset_name]["acc_5"] = acc_5
                best_params_dict[dataset_name]["acc_5"] = param_dict
            with codecs.open(grid_search_log_path, 'a+', encoding="utf-8") as log_file:
                log_file.write(f"{s}\n")
            logging.info(f"Dataset: {dataset_name}, Acc@1: {acc_1}, Acc@5 : {acc_5}")

    for dataset_name in best_accs_dict.keys():
        logging.info(f"DATASET {dataset_name}")
        best_param_dict_acc_1 = best_params_dict[dataset_name]["acc_1"]
        best_param_dict_acc_5 = best_params_dict[dataset_name]["acc_5"]

        best_acc_1 = best_accs_dict[dataset_name]["acc_1"]
        best_acc_5 = best_accs_dict[dataset_name]["acc_5"]
        logging.info(f"BEST ACC@1 : {best_acc_1}")
        logging.info(f"BEST ACC@5 : {best_acc_5}")
        logging.info(f"BEST ACC@1 SETUP:")
        for k, v in best_param_dict_acc_1.items():
            logging.info(f"\t{k}={v}")
        logging.info(f"BEST ACC@5 SETUP:")
        for k, v in best_param_dict_acc_5.items():
            logging.info(f"\t{k}={v}")


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
