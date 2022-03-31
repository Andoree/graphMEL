import logging
import os
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, AutoModel

from graphmel.scripts.training.model import GraphSAGEOverBert, BertOverNode2Vec
from graphmel.scripts.utils.io import load_dict, save_encoder_from_checkpoint


def main(args):
    model_description_path = os.path.join(args.graph_model_dir, "model_description.tsv")
    model_parameters_dict = load_dict(path=model_description_path, )

    tokenizer = AutoTokenizer.from_pretrained(model_parameters_dict["text_encoder"])
    bert_encoder = AutoModel.from_pretrained(model_parameters_dict["text_encoder"])

    multigpu_flag = False
    if args.graph_network_architecture == "graphsage":
        num_layers = int(model_parameters_dict["graphsage_num_layers"])
        hidden_channels = int(model_parameters_dict["graphsage_num_channels"])
        graphsage_dropout = float(model_parameters_dict["graphsage_dropout"])
        model = GraphSAGEOverBert(bert_encoder=bert_encoder, hidden_channels=hidden_channels, num_layers=num_layers,
                                  graphsage_dropout=graphsage_dropout, multigpu_flag=multigpu_flag)
    elif args.graph_network_architecture == "node2vec":
        seq_max_length = 32
        model = BertOverNode2Vec(bert_encoder=bert_encoder, seq_max_length=seq_max_length, multigpu_flag=multigpu_flag)
    else:
        raise NotImplementedError(f"Graph architecture {args.graph_network_architecture} is not supported")

    checkpoint = torch.load(args.model_checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])

    save_encoder_from_checkpoint(graph_over_bert_model=model, bert_tokenizer=tokenizer, save_path=args.output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--graph_network_architecture', type=str)
    parser.add_argument('--model_checkpoint_path', type=str)
    parser.add_argument('--graph_model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()

    main(arguments)
