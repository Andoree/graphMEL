import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from graphmel.scripts.training.dataset import load_data_and_bert_model, Node2vecDataset


def debug_node2vec_dataloader(loader: DataLoader, tokenizer: transformers.PreTrainedTokenizer):
    # <batch_size, context, seq_length>

    for pos_batch_input_ids, pos_batch_attention_masks, neg_batch_input_ids, neg_batch_attention_masks in loader:
        pos_sample_input_ids = random.choice(pos_batch_input_ids)
        pos_tokens = tokenizer.batch_decode(pos_sample_input_ids)
        pos_tokens = [token.replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip() for token in pos_tokens]

        neg_sample_input_ids = random.choice(neg_batch_input_ids)
        neg_tokens = tokenizer.batch_decode(neg_sample_input_ids)
        neg_tokens = [token.replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip() for token in neg_tokens]
        print("Positive", pos_tokens)
        print("Negative", neg_tokens)
        print('-' * 15)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    bert_encoder, node_id2token_ids_dict, edge_index, _, _ = \
        load_data_and_bert_model(train_node2terms_path=args.node2terms_path,
                                 train_edges_path=args.edges_path,
                                 val_node2terms_path=args.node2terms_path,
                                 val_edges_path=args.edges_path, text_encoder_name=args.text_encoder,
                                 text_encoder_seq_length=32, drop_relations_info=True)
    num_nodes = len(set(node_id2token_ids_dict.keys()))

    logging.info(f"There are {num_nodes} nodes")
    dataset = Node2vecDataset(edge_index=edge_index, node_id_to_token_ids_dict=node_id2token_ids_dict,
                              walk_length=10, walks_per_node=args.node2vec_train_walks_per_node, p=1, q=1,
                              num_negative_samples=1, context_size=7, num_nodes=num_nodes, seq_max_length=32)

    loader = DataLoader(dataset, collate_fn=dataset.sample, batch_size=32,
                        num_workers=args.dataloader_num_workers, shuffle=True)
    debug_node2vec_dataloader(loader, tokenizer=tokenizer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--node2terms_path', type=str)
    parser.add_argument('--edges_path', type=str)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--dataloader_num_workers', type=int)
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
