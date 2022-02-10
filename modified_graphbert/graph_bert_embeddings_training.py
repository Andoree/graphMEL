import datetime
import logging
import os
import sys

from graph_bert.DatasetLoader import DatasetLoader
from helpers.create_dataset import DatasetCreator
from helpers.logging_helpers import Writer, ExitOnExceptionHandler
from helpers.parse_args import load_args
from helpers.preprocess import Preprocessor
from helpers.pretrain import GraphBertNodeAttributeReconstruction
from graph_bert.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
from graph_bert.MethodGraphBertNodeConstruct import MethodGraphBertNodeConstruct
from graph_bert_embeddings_predict import GraphBERTEmbeddingsSaver

def main():
    options = load_args()

    if options.global_parser is None:
        logging.critical("Please provide actions to perform!")

    dir_path = None

    if options.global_parser == 'generate':
        pos = options.pos
        is_directed = options.is_directed
        language = options.language
        ft_path = f"{options.fasttext_dir}/cc.{language}.300.bin"
        dir_path = f'{options.dir_path}/wordnet_{pos}_is_directed_{int(is_directed)}_{language}'

        dataset_creator = DatasetCreator(ft_path, dir_path)
        dataset_creator.create_dataset(pos, is_directed)

    # ----------------------------------------------
    # data loading
    # ----------------------------------------------

    if options.dataset_name is not None:
        dataset_name = options.dataset_name
    elif dir_path is not None:
        dataset_name = f"{os.path.split(dir_path)[-1]}_{options.type}"
    else:
        logging.critical("dataset name should be provided in your configuration")
        dataset_name = None

    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = f"{options.dir_path}/{dataset_name}/"
    data_obj.dataset_name = dataset_name
    data_obj.k = options.k
    data_obj.compute_s = True

    if options.use_cuda:
        data_obj.device = "cuda"

    # ----------------------------------------------
    # data preparation (optional)
    # ----------------------------------------------

    if options.global_parser == 'preprocess' or (hasattr(options, 'preprocess') and options.preprocess is not None):
        preprocessor = Preprocessor(data_obj=data_obj, dataset_name=dataset_name)
        preprocessor.run_wl()
        preprocessor.run_graph_batching(k_list=options.k_list)
        preprocessor.run_hop_distance(options.max_hop_k)

    # ----------------------------------------------
    # training (optional)
    # ----------------------------------------------

    if options.global_parser == 'pretrain' or (hasattr(options, 'pretrain') and options.pretrain is not None):
        hidden_size = intermediate_size = options.hidden_size

        pre_trained_path = f"{options.pretrained_path}/{dataset_name}/node_reconstruct_model/"

        data_obj.load_all_tag = True
        graph_bert = GraphBertNodeAttributeReconstruction(data_obj=data_obj, dataset_name=dataset_name,
                                                          pretrained_path=pre_trained_path)
        graph_bert.k = options.k
        graph_bert.nfeature = options.feature_size
        graph_bert.x_size = options.feature_size
        graph_bert.graph_size = options.graph_size
        graph_bert.lr = options.lr
        graph_bert.max_epoch = options.max_epoch
        graph_bert.residual_type = options.residual_type
        graph_bert.num_attention_heads = options.num_attention_heads
        graph_bert.num_hidden_layers = options.num_hidden_layers
        graph_bert.hidden_size = hidden_size
        graph_bert.intermediate_size = intermediate_size

        graph_bert.graphbert_reconstruct()
        graph_bert.graphbert_network_recovery()



def optuna_entry_point(params, data_obj, id2label, dataset_name, pre_trained_path):
    
    data_obj.load_all_tag = True
    graph_bert = GraphBertNodeAttributeReconstruction(data_obj=data_obj, dataset_name=dataset_name,
                                                      pretrained_path=pre_trained_path)
    graph_bert.k = params['k']
    graph_bert.nfeature = params['nfeature']
    graph_bert.x_size = params['x_size']
    graph_bert.graph_size = params['graph_size']
    graph_bert.lr = params['lr']
    graph_bert.max_epoch = params['max_epoch']
    graph_bert.residual_type = params['residual_type']
    graph_bert.num_attention_heads = params['num_attention_heads']
    graph_bert.num_hidden_layers = params['num_hidden_layers']
    graph_bert.hidden_size = params['hidden_size']
    graph_bert.intermediate_size = params['intermediate_size']

    graph_bert.graphbert_reconstruct()
    graph_bert.graphbert_network_recovery()

    data_obj = data_obj.load()
    data = (data_obj['raw_embeddings'], data_obj['wl_embedding'], data_obj['hop_embeddings'],
               data_obj['int_embeddings'])

    gbes = GraphBERTEmbeddingsSaver("", model=MethodGraphBertNodeConstruct, x_size=graph_bert.x_size,
                                    intermediate_size=graph_bert.intermediate_size,
                                    num_attention_heads=graph_bert.num_attention_heads,
                                    num_hidden_layers=graph_bert.num_hidden_layers, k=graph_bert.k,
                                    nfeature=graph_bert.nfeature)
    embs1 = gbes.compute_embeddings(data, data_obj["index_id_map"], id2label)
    
    gbes = GraphBERTEmbeddingsSaver("", model=MethodGraphBertGraphRecovery, x_size=graph_bert.x_size,
                                    intermediate_size=graph_bert.intermediate_size,
                                    num_attention_heads=graph_bert.num_attention_heads,
                                    num_hidden_layers=graph_bert.num_hidden_layers, k=graph_bert.k,
                                    nfeature=graph_bert.nfeature)
    embs2 = gbes.compute_embeddings(data, data_obj["index_id_map"], id2label)

    return embs1, embs2


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  ExitOnExceptionHandler(
                                      filename=f'graph-bert_training_{datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}.log')])
    sys.stdout = Writer(logging.info)
    sys.stderr = Writer(logging.error)
    main()
