import logging
import os
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, AutoModel

from graphmel.scripts.utils.io import load_dict, save_encoder_from_checkpoint


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_description_path = os.path.join(args.graph_model_dir, "model_description.tsv")
    model_parameters_dict = load_dict(path=model_description_path, )
    
    tokenizer = AutoTokenizer.from_pretrained(model_parameters_dict["text_encoder"])
    bert_encoder = AutoModel.from_pretrained(model_parameters_dict["text_encoder"]).to(device)

    checkpoint = torch.load(args.model_checkpoint_path)
    bert_encoder.load_state_dict(checkpoint["model_state"])

    save_encoder_from_checkpoint(bert_encoder=bert_encoder, bert_tokenizer=tokenizer, save_path=args.output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--model_checkpoint_path', type=str)
    parser.add_argument('--graph_model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()

    main(arguments)
