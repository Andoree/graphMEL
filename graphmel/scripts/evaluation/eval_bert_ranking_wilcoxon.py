from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

from utils import read_vocab, read_dataset
from target_bert_ranker import BERTRanker
from typing import List
from scipy.stats import wilcoxon

def check_label(predicted_cui: str, golden_cui: str) -> int:
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.replace('+', '|').split("|")).intersection(
        set(golden_cui.replace('+', '|').split("|")))) > 0)


def is_correct(meddra_code: str, candidates: List[str], topk: int = 1) -> int:
    for candidate in candidates[:topk]:
        if check_label(candidate, meddra_code): return 1
    return 0


def get_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model_dir', help='Path to the directory containing BERT checkpoint', type=str)
    parser.add_argument('--sapbert_dir', type=str)
    parser.add_argument('--coder_dir', type=str)
    parser.add_argument('--data_folder', help='Path to the directory containing BioSyn format dataset', type=str)
    parser.add_argument('--vocab', help='Path to the vocabulary file in BioSyn format', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    ################
    entities = read_dataset(args.data_folder)
    ################
    entity_texts = [e['entity_text'].lower() for e in entities]
    labels = [e['label'] for e in entities]
    ##################
    vocab = read_vocab(args.vocab)
    target_bert_ranker = BERTRanker(args.model_dir, vocab)

    predicted_target_model_labels = target_bert_ranker.predict(entity_texts)
    target_model_correct_top1 = []
    target_model_correct_top5 = []
    for label, predicted_top_labels in tqdm(zip(labels, predicted_target_model_labels), total=len(labels)):
        target_model_correct_top1.append(is_correct(label, predicted_top_labels, topk=1))
        target_model_correct_top5.append(is_correct(label, predicted_top_labels, topk=5))
    del target_bert_ranker
    print("Target model Acc@1 is ", np.mean(target_model_correct_top1))
    print("Target model Acc@5 is ", np.mean(target_model_correct_top5))

    sapbert_bert_ranker = BERTRanker(args.sapbert_dir, vocab)
    predicted_sapbert_labels = sapbert_bert_ranker.predict(entity_texts)
    sapbert_correct_top1 = []
    sapbert_correct_top5 = []
    for label, predicted_top_labels in tqdm(zip(labels, predicted_sapbert_labels), total=len(labels)):
        sapbert_correct_top1.append(is_correct(label, predicted_top_labels, topk=1))
        sapbert_correct_top5.append(is_correct(label, predicted_top_labels, topk=5))
    del sapbert_bert_ranker
    print("SapBERT model Acc@1 is ", np.mean(sapbert_correct_top1))
    print("SapBERT model Acc@5 is ", np.mean(sapbert_correct_top5))

    coder_bert_ranker = BERTRanker(args.coder_dir, vocab)
    predicted_coder_labels = coder_bert_ranker.predict(entity_texts)
    coder_correct_top1 = []
    coder_correct_top5 = []
    for label, predicted_top_labels in tqdm(zip(labels, predicted_coder_labels), total=len(labels)):
        coder_correct_top1.append(is_correct(label, predicted_top_labels, topk=1))
        coder_correct_top5.append(is_correct(label, predicted_top_labels, topk=5))
    del coder_bert_ranker
    print("CODER model Acc@1 is ", np.mean(coder_correct_top1))
    print("CODER model Acc@5 is ", np.mean(coder_correct_top5))

    res_target_model_sapbert_wilcoxon = wilcoxon(target_model_correct_top1, sapbert_correct_top1)
    res_target_model_coder_wilcoxon = wilcoxon(target_model_correct_top1, coder_correct_top1)
    print(f"Wilcoxon target model - SapBERT: {res_target_model_sapbert_wilcoxon}")
    print(f"Wilcoxon target model - CODER: {res_target_model_coder_wilcoxon}")
