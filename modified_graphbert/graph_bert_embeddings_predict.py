import argparse
import datetime
import logging
import os
import sys

import numpy as np
from nltk.corpus import wordnet as wn

from graph_bert.DatasetLoader import DatasetLoader
from graph_bert.MethodBertComp import GraphBertConfig
from graph_bert.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
from graph_bert.MethodGraphBertNodeConstruct import MethodGraphBertNodeConstruct
from helpers.logging_helpers import ExitOnExceptionHandler, Writer
from itertools import combinations
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_data(dataset_path, k, device):
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = '../data/' + dataset_path + '/'
    data_obj.dataset_name = dataset_path
    data_obj.k = k
    data_obj.device = device
    data_obj.load_all_tag = True
    return data_obj.load()


def get_query_embedding(word, final_embeddings, index_id_map):
    offset, definition = wn.synset(word).offset(), wn.synset(word).definition()
    index_of_synset = None

    for i, j in index_id_map.items():
        if j == offset:
            index_of_synset = i
            break

    query_embedding = final_embeddings[index_of_synset]
    return query_embedding


class GraphBERTEmbeddingsSaver:
    def __init__(self, model_name, model, x_size=300, device='cpu', max_index=116, intermediate_size=32,
                 num_attention_heads=2, num_hidden_layers=2, y_size=0, residual_type='graph_raw', k=5, nfeature=300):

        pretrained_path = './result/PreTrained_GraphBert/' + model_name
        bert_config = GraphBertConfig(residual_type=residual_type, k=k, x_size=x_size, y_size=y_size,
                                      hidden_size=intermediate_size, intermediate_size=intermediate_size,
                                      num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers,
                                      max_wl_role_index=max_index, max_hop_dis_index=max_index,
                                      max_inti_pos_index=max_index)

        self.model = model(bert_config, pretrained_path, device=device)
        self.model.eval()
        self.nfeature = nfeature

    def compute_and_save_embeddings(self, data, test_synsets, index_id_map, id2label, result_dir):
        final_embeddings = self.compute_embeddings(data, index_id_map, id2label)
        self.save_embeddings(test_synsets, final_embeddings, result_dir)

    def compute_embeddings(self, data, index_id_map, id2label):
        final_embeddings = np.zeros(shape=(len(index_id_map), self.nfeature), dtype=np.float32)

        for _index, raw_f, wl, init, hop in zip(index_id_map, *data):
            final_embeddings[_index, :] = np.array(
                self.model(raw_f.unsqueeze(0), wl.unsqueeze(0), init.unsqueeze(0), hop.unsqueeze(0))[0]
                    .cpu().detach())
        return self.get_embeddings_dict(final_embeddings, index_id_map, id2label)

    @staticmethod
    def get_embeddings_dict(embeddings, index2id_map, id2label):
        return {id2label[index]: embeddings[_id] for _id, index in index2id_map.items()}

    def save_embeddings(self, test_synsets, embeddings, result_dir):
        with open(os.path.join(result_dir, f"{self.model.__class__.__name__}_model_train_embeddings.txt"), 'w') as w1:
            with open(os.path.join(result_dir, f"{self.model.__class__.__name__}_model_test_embeddings.txt"),
                      'w') as w2:
                for synset_name, embedding in embeddings.items():
                    if synset_name in test_synsets:
                        text_embedding = " ".join([str(e) for e in embedding])
                        w2.write(f"{synset_name} {text_embedding}\n")
                    else:
                        text_embedding = " ".join([str(e) for e in embedding])
                        w1.write(f"{synset_name} {text_embedding}\n")


def main(options):
    logging.info("Loading data")
    loaded_data = load_data(options.dataset_name, options.k, options.device)
    dataset = (loaded_data['raw_embeddings'], loaded_data['wl_embedding'], loaded_data['hop_embeddings'],
               loaded_data['int_embeddings'])
    index_id_map = loaded_data['index_id_map']

    logging.info("Loading train dataset indices")
    idx_features_labels = np.genfromtxt("{}/node".format('../data/' + dataset_name + '/'), dtype=np.dtype(str))
    id2label = {int(i): j for i, j in zip(idx_features_labels[:, 0], idx_features_labels[:, -1])}

    logging.info("Loading test set names")
    with open("test_synsets.txt") as f:
        test_synset_names = f.read().split("\n")

    logging.info("Loading saver")
    saver = GraphBERTEmbeddingsSaver(options.model_path, options.model_type)

    logging.info("Computing and saving embeddings")
    saver.compute_and_save_embeddings(dataset, test_synset_names, index_id_map, id2label, options.save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  ExitOnExceptionHandler(
                                      filename=f'graph-bert_predict_{datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}.log')])
    sys.stdout = Writer(logging.info)
    sys.stderr = Writer(logging.error)

    dataset_name = "old_wordnet_n_is_directed_1_en_synsets"
    model_dataset_name = "wordnet_n_is_directed_1_en_synsets"

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset_name", help="device: cpu or cuda",
                        default='old_wordnet_n_is_directed_1_en_synsets')
    parser.add_argument('-m', "--model_path", help="device: cpu or cuda",
                        default='wordnet_n_is_directed_1_en_synsets/node_reconstruct_model')
    parser.add_argument('-t', "--model_type",
                        type=lambda x: MethodGraphBertNodeConstruct if x == '1' else MethodGraphBertGraphRecovery)
    parser.add_argument('-s', '--save_path', help="path to save embeddings", default="../data/")
    parser.add_argument('-k', '--k', help="k to test", default=5)
    parser.add_argument('--device', help="device: cpu or cuda", default='cpu')

    main(parser.parse_args())
