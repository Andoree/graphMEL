import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from graphmel.scripts.evaluation.bert_ranker import BERTRanker
from graphmel.scripts.evaluation.eval_bert_ranking import is_correct
from graphmel.scripts.evaluation.evaluate_all_checkpoints_in_dir import evaluate_single_checkpoint_acc1_acc5
from graphmel.scripts.evaluation.utils import read_vocab, read_dataset


def main(args):
    output_evaluation_file_path = args.output_evaluation_file_path
    output_dir = os.path.dirname(output_evaluation_file_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    ##################
    vocab = read_vocab(args.vocab)
    evaluation_results = []
    for model_setup_name in os.listdir(args.input_model_setups_dir):
        model_setup_dir = os.path.join(args.input_model_setups_dir, model_setup_name)
        logging.info(f"Processing setup directory: {model_setup_dir}")
        if os.path.isdir(model_setup_dir):
            for checkpoint_name in os.listdir(model_setup_dir):
                checkpoint_dir = os.path.join(model_setup_dir, checkpoint_name)
                if os.path.isdir(checkpoint_dir):
                    for dataset_name in args.data_folder:
                        dataset_path = os.path.join(args.data_folder, dataset_name)
                        entities = read_dataset(dataset_path)
                        entity_texts = [e['entity_text'].lower() for e in entities]
                        labels = [e['label'] for e in entities]
                        logging.info(f"Processing checkpoint: {checkpoint_dir}")
                        acc_1, acc_5 = evaluate_single_checkpoint_acc1_acc5(checkpoint_path=checkpoint_dir, vocab=vocab,
                                                                            entity_texts=entity_texts, labels=labels)
                        evaluation_dict = {"Model setup": model_setup_name, "dataset_name": dataset_name,
                                           "Checkpoint name": checkpoint_name, "Acc@1": acc_1, "Acc@5": acc_5}
                        evaluation_results.append(evaluation_dict)
                        logging.info(
                            f"Finished processing checkpoint: {checkpoint_dir}, dataset: {dataset_name},"
                            f" Acc@1: {acc_1}, Acc@5 : {acc_5}")
                else:
                    pass
        else:
            pass
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(output_evaluation_file_path, sep='\t', index=False, encoding="utf-8")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset_names', type=str, nargs='+')
    parser.add_argument('--vocab', help='Path to the vocabulary file in BioSyn format', type=str)
    parser.add_argument('--input_model_setups_dir', type=str)
    parser.add_argument('--output_evaluation_file_path', type=str)
    arguments = parser.parse_args()

    main(arguments)
