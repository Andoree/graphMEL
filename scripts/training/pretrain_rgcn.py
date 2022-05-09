import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphmel.scripts.training.data.data_utils import create_rel_id2inverse_rel_id_map
from graphmel.scripts.training.data.relation_dataset import RelationalNeighborSampler
from graphmel.scripts.training.data.dataset import convert_edges_tuples_to_oriented_edge_index_with_relations, \
    load_data_and_bert_model, SimpleDataset
from graphmel.scripts.training.model import RGCNLinkPredictorOverBert
from graphmel.scripts.training.training import train_model
from graphmel.scripts.utils.io import save_dict, load_dict


def rgcn_step(model, batch, reg_lambda, loss_fn, device):
    (pos_src_batch_size, pos_src_adjs, pos_src_input_ids, pos_src_att_masks, pos_src_rel_ids) = batch["pos_src_input"]
    (pos_trg_batch_size, pos_trg_adjs, pos_trg_input_ids, pos_trg_att_masks, pos_trg_rel_ids) = batch["pos_trg_input"]
    (neg_src_batch_size, neg_src_adjs, neg_src_input_ids, neg_src_att_masks, neg_src_rel_ids) = batch["neg_src_input"]
    (neg_trg_batch_size, neg_trg_adjs, neg_trg_input_ids, neg_trg_att_masks, neg_trg_rel_ids) = batch["neg_trg_input"]
    rel_ids,  batch_size = batch["rel_id"].to(device), batch["batch_size"]

    pos_src_adjs, pos_src_input_ids, pos_src_att_masks, pos_src_rel_ids = \
        pos_src_adjs.to(device), pos_src_input_ids.to(device), pos_src_att_masks.to(device), pos_src_rel_ids.to(device)
    pos_trg_adjs, pos_trg_input_ids, pos_trg_att_masks, pos_trg_rel_ids = \
        pos_trg_adjs.to(device), pos_trg_input_ids.to(device), pos_trg_att_masks.to(device), pos_trg_rel_ids.to(device)
    neg_src_adjs, neg_src_input_ids, neg_src_att_masks, neg_src_rel_ids = \
        neg_src_adjs.to(device), neg_src_input_ids.to(device), neg_src_att_masks.to(device), neg_src_rel_ids.to(device)
    neg_trg_adjs, neg_trg_input_ids, neg_trg_att_masks, neg_trg_rel_ids = \
        neg_trg_adjs.to(device), neg_trg_input_ids.to(device), neg_trg_att_masks.to(device), neg_trg_rel_ids.to(device)


    pos_scores, pos_reg, neg_scores, neg_reg = \
        model(pos_src_input_ids=pos_src_input_ids, pos_src_attention_mask=pos_src_att_masks, pos_src_adjs=pos_src_adjs,
              pos_src_rel_ids=pos_src_rel_ids, pos_trg_rel_ids=pos_trg_rel_ids, neg_src_rel_ids=neg_src_rel_ids,
              neg_trg_rel_ids=neg_trg_rel_ids,
              pos_trg_input_ids=pos_trg_input_ids, pos_trg_attention_mask=pos_trg_att_masks, pos_trg_adjs=pos_trg_adjs,
              neg_src_input_ids=neg_src_input_ids, neg_src_attention_mask=neg_src_att_masks, neg_src_adjs=neg_src_adjs,
              neg_trg_input_ids=neg_trg_input_ids, neg_trg_attention_mask=neg_trg_att_masks, neg_trg_adjs=neg_trg_adjs,
              rel_ids=rel_ids, batch_size=batch_size)
    pos_labels = torch.ones((rel_ids.size()[0], 1), dtype=torch.float, device=device)
    neg_labels = torch.zeros((rel_ids.size()[0], 1), dtype=torch.float, device=device)
    true_labels = torch.cat([pos_labels, neg_labels], dim=0).view(-1).to(device)
    pred_scores = torch.cat([pos_scores, neg_scores], dim=0).view(-1)
    pos_reg, neg_reg = pos_reg.to(device), neg_reg.to(device)

    loss = loss_fn(pred_scores, true_labels) + reg_lambda * (pos_reg + neg_reg)

    return loss


def rgcn_train_epoch(model: RGCNLinkPredictorOverBert, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                     loss_fn, reg_lambda: float, device):
    model.train()
    total_loss = 0
    num_steps = 0
    # pbar = tqdm(train_loader, miniters=len(train_loader) // 10000, total=len(train_loader) // train_loader.batch_size)
    for batch in tqdm(train_loader, miniters=len(train_loader) // 10000, total=len(train_loader)):
        optimizer.zero_grad()
        loss = rgcn_step(model=model, batch=batch, reg_lambda=reg_lambda,
                         loss_fn=loss_fn, device=device)
        loss.backward()
        optimizer.step()
        num_steps += 1
        total_loss += float(loss)
        # pbar.set_description(f"Loss: {loss}", refresh=True)
    return total_loss / len(train_loader), num_steps


def rgcn_val_epoch(model: RGCNLinkPredictorOverBert, val_loader: DataLoader,
                   loss_fn, reg_lambda: float, device):
    model.eval()
    total_loss = 0
    # pbar = tqdm(val_loader, miniters=len(val_loader) // 10000, total=len(val_loader) // val_loader.batch_size)
    with torch.no_grad():
        for batch in tqdm(val_loader, miniters=len(val_loader) // 10000, total=len(val_loader)):
            loss = rgcn_step(model=model, batch=batch, reg_lambda=reg_lambda,
                             loss_fn=loss_fn, device=device)
            total_loss += float(loss)
            # pbar.set_description(f"Loss: {loss}", refresh=True)
    return total_loss / len(val_loader)


def main(args):
    output_dir = args.output_dir
    conv_type = "fast_rgcn_conv" if args.rgcn_use_fast_conv else "rgcn_conv"
    output_subdir = f"nhc-nl-nba-nbl_{'.'.join((str(x) for x in args.rgcn_num_hidden_channels))}-{args.rgcn_num_layers}-" \
                    f"{args.rgcn_num_bases}-{args.rgcn_num_blocks}_{args.use_rel_or_rela}_reg-{args.distmult_l2_reg_lambda}_" \
                    f"lr_{args.learning_rate}_b_{args.batch_size}_{conv_type}"
    output_dir = os.path.join(output_dir, output_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    train_node2terms_path = os.path.join(args.train_dir, "node_id2terms_list")
    train_edges_path = os.path.join(args.train_dir, "edges")
    train_rel2id_path = os.path.join(args.train_dir, "rel2id")
    train_rela2id_path = os.path.join(args.train_dir, "rela2id")

    val_node2terms_path = os.path.join(args.val_dir, "node_id2terms_list")
    val_edges_path = os.path.join(args.val_dir, "edges")
    val_rel2id_path = os.path.join(args.val_dir, "rel2id")
    val_rela2id_path = os.path.join(args.val_dir, "rela2id")

    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )

    bert_encoder, train_node_id2token_ids_dict, train_edges_tuples, val_node_id2token_ids_dict, val_edges_tuples = \
        load_data_and_bert_model(train_node2terms_path=train_node2terms_path,
                                 train_edges_path=train_edges_path,
                                 val_node2terms_path=val_node2terms_path,
                                 val_edges_path=val_edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=args.text_encoder_seq_length, drop_relations_info=False)

    train_num_nodes = len(set(train_node_id2token_ids_dict.keys()))
    val_num_nodes = len(set(val_node_id2token_ids_dict.keys()))

    train_rel2id = load_dict(train_rel2id_path)
    train_rela2id = load_dict(train_rela2id_path)
    val_rel2id = load_dict(val_rel2id_path)
    val_rela2id = load_dict(val_rela2id_path)
    assert (len(train_rel2id.keys()) == len(val_rel2id.keys())) and (
            len(train_rela2id.keys()) == len(val_rela2id.keys()))
    for key in train_rel2id.keys():
        assert train_rel2id[key] == val_rel2id[key]
    for key in train_rela2id.keys():
        assert train_rela2id[key] == val_rela2id[key]
    if args.use_rel_or_rela == "rel":
        # TODO: Потом если надо поменять
        # num_relations = len(train_rel2id.keys()) * 2 - 1
        num_relations = len(train_rel2id.keys())
    elif args.use_rel_or_rela == "rela":
        # num_relations = len(train_rela2id.keys()) * 2 - 1
        num_relations = len(train_rela2id.keys())
        train_rel2id = train_rela2id
    else:
        raise ValueError(f"Invalid 'use_rel_or_rela' parameter: {args.use_rel_or_rela}")

    rel2id = {k: int(v) for k, v in train_rel2id.items()}
    rel_id2inverse_rel_id = create_rel_id2inverse_rel_id_map(rel2id=rel2id)

    train_edge_index, train_edge_rel_ids = \
        convert_edges_tuples_to_oriented_edge_index_with_relations(train_edges_tuples, args.use_rel_or_rela)
    assert train_edge_index.size()[1] == len(train_edge_rel_ids)
    val_edge_index, val_edge_rel_ids = \
        convert_edges_tuples_to_oriented_edge_index_with_relations(val_edges_tuples, args.use_rel_or_rela)
    assert val_edge_index.size()[1] == len(val_edge_rel_ids)
    train_num_edges = train_edge_index.size()[1]
    val_num_edges = val_edge_index.size()[1]

    train_edge_idx = torch.LongTensor(range(train_num_edges))
    val_edge_idx = torch.LongTensor(range(val_num_edges))

    train_loader = RelationalNeighborSampler(edge_index=train_edge_index,
                                             node_id_to_token_ids_dict=train_node_id2token_ids_dict,
                                             rel_ids=train_edge_rel_ids, batch_size=args.batch_size,
                                             node_neighborhood_sizes=args.node_neighborhood_sizes,
                                             rel_id2inverse_rel_id=rel_id2inverse_rel_id, node_idx=train_edge_idx,
                                             num_nodes=train_num_nodes, seq_max_length=args.text_encoder_seq_length,
                                             num_workers=args.dataloader_num_workers, shuffle=True)
    val_loader = RelationalNeighborSampler(edge_index=val_edge_index,
                                           node_id_to_token_ids_dict=val_node_id2token_ids_dict,
                                           rel_ids=val_edge_rel_ids, batch_size=args.batch_size,
                                           node_neighborhood_sizes=args.node_neighborhood_sizes,
                                           rel_id2inverse_rel_id=rel_id2inverse_rel_id, node_idx=val_edge_idx,
                                           num_nodes=val_num_nodes, seq_max_length=args.text_encoder_seq_length,
                                           num_workers=args.dataloader_num_workers, shuffle=False, drop_last=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multigpu_flag = False
    if args.gpus > 1:
        multigpu_flag = True

    model = RGCNLinkPredictorOverBert(bert_encoder=bert_encoder, num_hidden_channels=args.rgcn_num_hidden_channels,
                                      num_layers=args.rgcn_num_layers, num_relations=num_relations,
                                      num_bases=args.rgcn_num_bases, num_blocks=args.rgcn_num_blocks,
                                      use_fast_conv=args.rgcn_use_fast_conv, multigpu_flag=multigpu_flag).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    train_model(model=model, train_epoch_fn=rgcn_train_epoch, val_epoch_fn=rgcn_val_epoch,
                chkpnt_path=args.model_checkpoint_path, train_loader=train_loader, val_loader=val_loader,
                learning_rate=args.learning_rate, num_epochs=args.num_epochs, output_dir=output_dir,
                save_chkpnt_epoch_interval=args.save_every_N_epoch, device=device, loss_fn=loss_fn,
                reg_lambda=args.distmult_l2_reg_lambda, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()

    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=False)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--text_encoder_seq_length', type=int)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--save_every_N_epoch', type=int, default=1)
    parser.add_argument('--rgcn_num_hidden_channels', type=int, nargs='+')
    parser.add_argument('--rgcn_num_layers', type=int)
    parser.add_argument('--rgcn_num_bases', type=int)
    parser.add_argument('--rgcn_num_blocks', type=int, )
    parser.add_argument('--rgcn_use_fast_conv', action="store_true")
    parser.add_argument('--node_neighborhood_sizes', type=int, nargs='+')
    parser.add_argument('--use_rel_or_rela', type=str, choices=['rel', 'rela', ])
    parser.add_argument('--distmult_l2_reg_lambda', type=float, )
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
