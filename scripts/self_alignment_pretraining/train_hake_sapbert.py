#!/usr/bin/env python
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from graphmel.scripts.utils.umls2graph import filter_transitive_hierarchical_relations, \
    filter_hierarchical_semantic_type_nodes
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphmel.scripts.models.hake_sapbert import HakeSapMetricLearning
from graphmel.scripts.self_alignment_pretraining.dataset import SapMetricLearningHierarchicalDataset
from graphmel.scripts.self_alignment_pretraining.sapbert_training import train_graph_sapbert_model
from graphmel.scripts.training.data.dataset import load_positive_pairs, map_terms2term_id, \
    create_term_id2tokenizer_output, load_tree_dataset_and_bert_model
from graphmel.scripts.utils.io import save_dict, read_mrsty, load_node_id2terms_list


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

    # HAKE configuration
    parser.add_argument('--negative_sample_size', type=int)
    parser.add_argument('--hake_gamma', type=float)
    parser.add_argument('--hake_modulus_weight', type=float)
    parser.add_argument('--hake_phase_weight', type=float)
    parser.add_argument('--hake_adversarial_temperature', type=float)
    parser.add_argument('--hake_loss_weight', type=float)
    parser.add_argument('--filter_transitive_relations', action="store_true")
    parser.add_argument('--filter_semantic_type_nodes', action="store_true")
    # parser.add_argument('--node_id_lower_bound_filtering', type=int, required=False)
    parser.add_argument('--mrsty', type=str, required=False)

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


def hake_sapbert_train_step(model: HakeSapMetricLearning, batch, amp, device):
    term_1_input = (batch["term_1_input_ids"].to(device), batch["term_1_att_mask"].to(device))
    term_2_input = (batch["term_2_input_ids"].to(device), batch["term_2_att_mask"].to(device))
    concept_ids, rel_ids = batch["anchor_concept_id"].to(device), batch["rel_id"].to(device)

    pos_parent_input = (batch["pos_parent_input_ids"].to(device), batch["pos_parent_att_mask"].to(device))
    pos_child_input = (batch["pos_child_input_ids"].to(device), batch["pos_child_att_mask"].to(device))
    neg_rel_corr_h_input = (batch["neg_rel_corr_h_input_ids"].to(device), batch["neg_rel_corr_h_att_mask"].to(device))
    neg_rel_corr_t_input = (batch["neg_rel_corr_t_input_ids"].to(device), batch["neg_rel_corr_t_att_mask"].to(device))
    sample_weight = batch["sample_weight"].to(device)

    if amp:
        with autocast():
            sapbert_loss, hake_loss = model(term_1_input=term_1_input, term_2_input=term_2_input,
                                            concept_ids=concept_ids, model_mode="train",
                                            pos_parent_input=pos_parent_input,
                                            neg_rel_corr_h_input=neg_rel_corr_h_input,
                                            neg_rel_corr_t_input=neg_rel_corr_t_input,
                                            pos_child_input=pos_child_input,
                                            sample_weight=sample_weight, rel_ids=rel_ids)

    else:
        sapbert_loss, hake_loss = model(term_1_input=term_1_input, term_2_input=term_2_input,
                                        concept_ids=concept_ids, model_mode="train",
                                        pos_parent_input=pos_parent_input,
                                        neg_rel_corr_h_input=neg_rel_corr_h_input,
                                        neg_rel_corr_t_input=neg_rel_corr_t_input,
                                        pos_child_input=pos_child_input,
                                        sample_weight=sample_weight, rel_ids=rel_ids)

    return sapbert_loss, hake_loss


def hake_sapbert_eval_step(model: HakeSapMetricLearning, batch, amp, device, ):
    term_1_input = (batch["term_1_input_ids"].to(device), batch["term_1_att_mask"].to(device))
    term_2_input = (batch["term_2_input_ids"].to(device), batch["term_2_att_mask"].to(device))
    concept_ids = batch["anchor_concept_id"].to(device)

    if amp:
        with autocast():
            sapbert_loss = model(term_1_input=term_1_input, term_2_input=term_2_input,
                                 concept_ids=concept_ids, model_mode="validation", )

    else:
        sapbert_loss = model(term_1_input=term_1_input, term_2_input=term_2_input,
                             concept_ids=concept_ids, model_mode="validation", )
    return sapbert_loss


def train_hake_sapbert(model: HakeSapMetricLearning, train_loader: SapMetricLearningHierarchicalDataset,
                       hake_loss_weight, optimizer: torch.optim.Optimizer, scaler, amp, device, **kwargs):
    model.train()
    total_loss = 0
    total_sapbert_loss = 0
    total_hake_loss = 0
    num_steps = 0
    for batch in tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader)):
        optimizer.zero_grad()
        sapbert_loss, hake_loss = hake_sapbert_train_step(model=model, batch=batch, amp=amp, device=device)

        loss = sapbert_loss + hake_loss_weight * hake_loss
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        num_steps += 1
        total_loss += float(loss)
        total_sapbert_loss += float(sapbert_loss)
        total_hake_loss += float(hake_loss)
        # wandb.log({"Train loss": loss.item()})
    total_loss /= (num_steps + 1e-9)
    total_sapbert_loss /= (num_steps + 1e-9)
    total_hake_loss /= (num_steps + 1e-9)
    logging.info(
        f"Finished HAKE-SAPBert training epoch. Weighted loss: {total_loss}, sapbert_loss: {total_sapbert_loss},"
        f"total_hake_loss: {total_hake_loss}")
    return total_loss, num_steps


def val_hake_sapbert(model: HakeSapMetricLearning, val_loader: SapMetricLearningHierarchicalDataset,
                     amp, device, **kwargs):
    model.eval()
    total_loss = 0
    num_steps = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, miniters=len(val_loader) // 100, total=len(val_loader)):
            sapbert_loss = hake_sapbert_eval_step(model=model, batch=batch, amp=amp, device=device)
            num_steps += 1
            total_loss += float(sapbert_loss)

    total_loss /= (num_steps + 1e-9)
    return total_loss


def main(args):
    print(args)
    output_dir = args.output_dir
    output_subdir = f"neg_{args.negative_sample_size}_gamma_{args.hake_gamma}_mw_" \
                    f"{args.hake_modulus_weight}_pw_{args.hake_phase_weight}_adv_{args.hake_adversarial_temperature}_" \
                    f"filt_trans_{args.filter_transitive_relations}_" \
                    f"filt_sem_type_{args.filter_semantic_type_nodes}_" \
                    f"hake_weight_{args.hake_loss_weight}_lr_{args.learning_rate}_b_{args.batch_size}"
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
    parent_children_adjacency_list_path = os.path.join(args.train_dir, "parent_childs_adjacency_list")
    child_parents_adjacency_list_path = os.path.join(args.train_dir, "child_parents_adjacency_list")

    bert_encoder, bert_tokenizer, node_id2token_ids_dict, parent_children_adjacency_list, child_parents_adjacency_list = \
        load_tree_dataset_and_bert_model(node2terms_path=node2terms_path, text_encoder_name=args.text_encoder,
                                         parent_children_adjacency_list_path=parent_children_adjacency_list_path,
                                         child_parents_adjacency_list_path=child_parents_adjacency_list_path,
                                         text_encoder_seq_length=args.max_length, )

    node_id2terms = load_node_id2terms_list(dict_path=node2terms_path, )

    if args.filter_semantic_type_nodes:
        mrsty_df = read_mrsty(args.mrsty)

        excluded_node_ids = filter_hierarchical_semantic_type_nodes(
            node_id2children=parent_children_adjacency_list,
            node_id2parents=child_parents_adjacency_list,
            node_id2terms=node_id2terms,
            mrsty_df=mrsty_df)

        # node_id2token_ids_dict = {node_id: token_ids for node_id, token_ids in
        #                                          node_id2token_ids_dict.items()
        #                                          if node_id not in excluded_node_ids}
        del mrsty_df

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
    del train_term2id

    train_dataset = SapMetricLearningHierarchicalDataset(pos_pairs_term_1_id_list=train_pos_pairs_term_1_id_list,
                                                         pos_pairs_term_2_id_list=train_pos_pairs_term_2_id_list,
                                                         pos_pairs_concept_ids_list=train_pos_pairs_concept_ids,
                                                         term_id2tokenizer_output=train_term_id2tok_out,
                                                         node_id2token_ids_dict=node_id2token_ids_dict,
                                                         concept_id2parents=child_parents_adjacency_list,
                                                         concept_id2childs=parent_children_adjacency_list,
                                                         seq_max_length=args.max_length,
                                                         negative_sample_size=args.negative_sample_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workers, collate_fn=train_dataset.collate_fn)

    val_dataloader = None
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
        del val_term2id

        val_dataset = SapMetricLearningHierarchicalDataset(pos_pairs_term_1_id_list=val_pos_pairs_term_1_id_list,
                                                           pos_pairs_term_2_id_list=val_pos_pairs_term_2_id_list,
                                                           pos_pairs_concept_ids_list=val_pos_pairs_concept_ids,
                                                           term_id2tokenizer_output=val_term_id2tok_out,
                                                           node_id2token_ids_dict=node_id2token_ids_dict,
                                                           concept_id2parents=child_parents_adjacency_list,
                                                           concept_id2childs=parent_children_adjacency_list,
                                                           seq_max_length=args.max_length,
                                                           negative_sample_size=args.negative_sample_size)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.dataloader_num_workers, collate_fn=val_dataset.collate_fn)

        val_epoch_fn = val_hake_sapbert
    device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    model = HakeSapMetricLearning(bert_encoder, hake_gamma=args.hake_gamma,
                                  hake_modulus_weight=args.hake_modulus_weight,
                                  hake_phase_weight=args.hake_phase_weight, num_relation=2, use_cuda=args.use_cuda,
                                  hake_adversarial_temperature=args.hake_adversarial_temperature,
                                  loss=args.loss, multigpu_flag=args.parallel, use_miner=args.use_miner,
                                  miner_margin=args.miner_margin, type_of_triplets=args.type_of_triplets,
                                  agg_mode=args.agg_mode, ).to(device)

    start = time.time()
    train_graph_sapbert_model(model=model, train_epoch_fn=train_hake_sapbert, val_epoch_fn=val_epoch_fn,
                              train_loader=train_dataloader,
                              val_loader=val_dataloader, hake_loss_weight=args.hake_loss_weight,
                              learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                              num_epochs=args.num_epochs, output_dir=output_dir,
                              save_chkpnt_epoch_interval=args.save_every_N_epoch, parallel=args.parallel,
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
