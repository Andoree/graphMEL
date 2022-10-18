import logging
import os
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, AutoModel

from graphmel.scripts.utils.io import load_dict, save_encoder_from_checkpoint, save_dict


def save_bert_encoder_from_graph_model(pretrained_graph_model_dir: str, bert_initialization_model: str,
                                       checkpoint_name: str, output_dir: str, device: str):
    model_description_path = os.path.join(pretrained_graph_model_dir, "model_description.tsv")
    model_checkpoint_path = os.path.join(pretrained_graph_model_dir, checkpoint_name)
    model_parameters_dict = load_dict(path=model_description_path, )
    pretrained_model_encoder_path_or_name = os.path.basename(model_parameters_dict["text_encoder"])
    if pretrained_model_encoder_path_or_name not in bert_initialization_model:
        raise ValueError(f"Pretrained graph-based encoder does not match with initialized model: "
                         f"{pretrained_model_encoder_path_or_name} not in {bert_initialization_model}")

    tokenizer = AutoTokenizer.from_pretrained(bert_initialization_model)
    bert_encoder = AutoModel.from_pretrained(bert_initialization_model).to(device)

    checkpoint = torch.load(model_checkpoint_path)
    bert_encoder.load_state_dict(checkpoint["model_state"])

    save_encoder_from_checkpoint(bert_encoder=bert_encoder, bert_tokenizer=tokenizer, save_path=output_dir)
    save_dict(save_path=os.path.join(output_dir, "model_description.tsv"), dictionary=model_parameters_dict)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_dir in os.listdir(args.input_pretrained_graph_models_dir):
        input_pretrained_graph_model_dir = os.path.join(args.input_pretrained_graph_models_dir, model_dir)
        for file in os.listdir(input_pretrained_graph_model_dir):
            if file.endswith('pth'):
                output_pretrained_bert_dir = os.path.join(args.output_dir, model_dir, f"{file}/")
                checkpoint_name = file
                if not os.path.exists(output_pretrained_bert_dir):
                    os.makedirs(output_pretrained_bert_dir)

                save_bert_encoder_from_graph_model(pretrained_graph_model_dir=input_pretrained_graph_model_dir,
                                                   bert_initialization_model=args.bert_initialization_model,
                                                   checkpoint_name=checkpoint_name,
                                                   output_dir=output_pretrained_bert_dir, device=device)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_pretrained_graph_models_dir', type=str)
    parser.add_argument('--bert_initialization_model', type=str)
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()

    main(arguments)
