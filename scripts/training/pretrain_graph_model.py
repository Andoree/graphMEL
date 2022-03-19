import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from graphmel.scripts.training.dataset import tokenize_node_terms, NeighborSampler, convert_edges_tuples_to_edge_index
from graphmel.scripts.training.model import GraphSAGEOverBert
from graphmel.scripts.utils.io import load_node_id2terms_list, load_tuples, update_log_file, save_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.data import NeighborSampler as RawNeighborSampler
from transformers import AutoTokenizer
from transformers import AutoModel


def model_step(model, input_ids, attention_mask, adjs, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model_output = model(input_ids, attention_mask, adjs)
    model_output, pos_out, neg_out = model_output.split(model_output.size(0) // 3, dim=0)

    pos_loss = F.logsigmoid((model_output * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(model_output * neg_out).sum(-1)).mean()
    loss = -pos_loss - neg_loss

    return model_output, loss


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_steps = 0

    for (batch_size, n_id, adjs, input_ids, attention_mask) in tqdm(train_loader, miniters=len(train_loader) // 100):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        model_output, loss = model_step(model, input_ids, attention_mask, adjs, device)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * model_output.size(0)
        num_steps += 1

    return total_loss / len(train_loader), num_steps


def eval_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (batch_size, n_id, adjs, input_ids, attention_mask) in tqdm(val_loader, miniters=len(val_loader) // 100):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            model_output, loss = model_step(model, input_ids, attention_mask, adjs, device)
            total_loss += float(loss) * model_output.size(0)

    return total_loss / len(val_loader)


def train_model(model, chkpnt_path: str, train_loader, val_loader, learning_rate: float, num_epochs: int,
                output_dir: str, device: torch.device):
    if chkpnt_path is not None:
        logging.info(f"Successfully loaded checkpoint from: {chkpnt_path}")
        checkpoint = torch.load(chkpnt_path)
        optimizer = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])

    else:
        start_epoch = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log_file_path = os.path.join(output_dir, "training_log.txt")

    train_loss_history = []
    val_loss_history = []
    logging.info("Starting training process....")
    global_num_steps = 0
    for i in range(start_epoch, start_epoch + num_epochs):
        epoch_train_loss, num_steps = train_epoch(model=model, train_loader=train_loader, optimizer=optimizer,
                                                  device=device)
        global_num_steps += num_steps
        epoch_val_loss_1 = eval_epoch(model=model, val_loader=val_loader, device=device)
        epoch_val_loss_2 = eval_epoch(model=model, val_loader=val_loader, device=device)
        # assert epoch_val_loss_1 == epoch_val_loss_2
        log_dict = {"epoch": i, "train loss": {epoch_train_loss}, "val loss 1": epoch_val_loss_1,
                    "val loss 2": epoch_val_loss_2}
        logging.info(', '.join((f"{k}: {v}" for k, v in log_dict.items())))
        # TODO: Потом убрать двойную проверку как удостоверюсь, что валидация детерминирована
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss_1)

        checkpoint = {
            'epoch': i + 1,
            'model_state': model.state_dict(),
            'optimizer': optimizer,
        }

        chkpnt_path = os.path.join(output_dir, f"checkpoint_e_{i}_steps_{global_num_steps}.pth")
        torch.save(checkpoint, chkpnt_path)
        # torch.save(model.state_dict(), chkpnt_path)
        update_log_file(path=log_file_path, dict_to_log=log_dict)


# TODO: Добавить логирование кривых обучения по эпохам
# TODO: Откусить испанский (или русский) для того, чтобы произвести на нём отладку
# TODO: Рисовать кривые обучения, надо же как-то затюнить параметры
def main(args):
    output_dir = args.output_dir
    output_subdir = f"gs_{args.graphsage_num_layers}-{args.graphsage_num_channels}_" \
                    f"{'.'.join((str(x) for x in args.graph_num_neighbors))}_{args.graphsage_dropout}_" \
                    f"lr_{args.learning_rate}_b_{args.batch_size}_rwl_{args.random_walk_length}"
    output_dir = os.path.join(output_dir, output_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=args, )

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
    train_loader = NeighborSampler(node_id_to_token_ids_dict=train_node_id2token_ids_dict, edge_index=train_edge_index,
                                   sizes=args.graph_num_neighbors, random_walk_length=args.random_walk_length,
                                   batch_size=args.batch_size,
                                   shuffle=True, num_nodes=train_num_nodes, seq_max_length=args.text_encoder_seq_length)
    val_loader = NeighborSampler(node_id_to_token_ids_dict=val_node_id2token_ids_dict, edge_index=val_edge_index,
                                 sizes=args.graph_num_neighbors, random_walk_length=args.random_walk_length,
                                 batch_size=args.batch_size,
                                 shuffle=False, num_nodes=val_num_nodes, seq_max_length=args.text_encoder_seq_length)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multigpu_flag = False
    if args.gpus > 1:
        multigpu_flag = True
    model = GraphSAGEOverBert(bert_encoder=bert_encoder, hidden_channels=args.graphsage_num_channels,
                              num_layers=args.graphsage_num_layers, multigpu_flag=multigpu_flag,
                              graphsage_dropout=args.graphsage_dropout).to(device)

    # model = nn.DataParallel(model)
    # model = model.to(device)
    train_model(model=model, chkpnt_path=args.model_checkpoint_path, train_loader=train_loader, val_loader=val_loader,
                learning_rate=args.learning_rate, num_epochs=args.num_epochs, output_dir=output_dir, device=device)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--train_node2terms_path', type=str)
    parser.add_argument('--train_edges_path', type=str)
    parser.add_argument('--val_node2terms_path', type=str)
    parser.add_argument('--val_edges_path', type=str)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--text_encoder_seq_length', type=int)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--graphsage_num_layers', type=int)
    parser.add_argument('--graphsage_num_channels', type=int)
    parser.add_argument('--graph_num_neighbors', type=int, nargs='+', )
    parser.add_argument('--graphsage_dropout', type=float, )
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
