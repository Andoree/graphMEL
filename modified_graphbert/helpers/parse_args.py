import argparse

import torch


def create_subparser_for_preprocess(parser):
    parser.add_argument('--k_list', help="list of k to generate data for training", nargs='+',
                                               type=lambda s: int(s),
                                               default=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50))
    parser.add_argument('--max_hop_k', default=10, type=int, help="from 1 to k hop steps")

    subparser = parser.add_subparsers(title="pretrain", dest='pretrain')
    subparser_pretrain = subparser.add_parser("pretrain")
    create_subparser_for_pretrain(subparser_pretrain)


def create_subparser_for_pretrain(subparser):
    subparser.add_argument("--feature_size", type=int, required=True)
    subparser.add_argument('--graph_size', type=int, required=True)
    subparser.add_argument('--lr', type=float, default=0.001)
    subparser.add_argument('--max_epoch', type=int, default=200)
    subparser.add_argument('--residual_type', default='graph_raw')
    subparser.add_argument('--num_attention_heads', type=int, default=2)
    subparser.add_argument('--num_hidden_layers', type=int, default=2)
    subparser.add_argument('--hidden_size', type=int, default=32)
    subparser.add_argument('--pretrained_path', default='../result/PreTrained_GraphBert/')


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--use_cuda", action="store_true", help="device: cpu or cuda")
    parser.add_argument('-k', type=int, default=5, help='k nearest nodes to use for training')
    parser.add_argument('-t', '--type', help="data level: synsets or lemmas", choices=['synsets', 'lemmas'],
                        default='synsets')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', "--dir_path", help="data path directory", default='../data')
    group.add_argument("-n", '--dataset_name', help="name of the generated dataset")

    sub_parser = parser.add_subparsers(dest='global_parser')

    # -----------------------------------------------------------------------------------------------------------------
    # main parser generate
    # -----------------------------------------------------------------------------------------------------------------
    parser_generate = sub_parser.add_parser("generate")
    parser_generate.add_argument("-p", '--pos', help="part-of-speech (nouns or verbs)", choices=['n', 'v'], default='n')
    parser_generate.add_argument('-d', "--is_directed", action="store_false", help="directed edged (default=True)")
    parser_generate.add_argument('-l', "--language", help="language of wordnet", default='en')
    parser_generate.add_argument('-f', "--fasttext_dir", help="path to fasttext pretrained model",
                                 default='../data/fasttext/')

    generate_subparser = parser_generate.add_subparsers(title="preprocess", dest='preprocess')
    generate_subparser_preprocess = generate_subparser.add_parser("preprocess")
    create_subparser_for_preprocess(generate_subparser_preprocess)

    # -----------------------------------------------------------------------------------------------------------------
    # main parser preprocess
    # -----------------------------------------------------------------------------------------------------------------

    parser_preprocess = sub_parser.add_parser("preprocess")
    create_subparser_for_preprocess(parser_preprocess)

    # -----------------------------------------------------------------------------------------------------------------
    # main parser pretrain
    # -----------------------------------------------------------------------------------------------------------------

    parser_pretrain = sub_parser.add_parser("pretrain")
    create_subparser_for_pretrain(parser_pretrain)

    # ----------------------------------------------
    # data generation
    # ----------------------------------------------
    return parser.parse_args()


def load_args_transformation():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--device", default='cuda', help="device: cpu or cuda")
    parser.add_argument('-s', "--src_embeddings_path", default="../../Graph-Bert/node_reconstruct_model_train_embeddings.txt")
    parser.add_argument('-o', "--out_path", default="../data/transformation_model")
    parser.add_argument('-n', "--node_path", default="../data/old_wordnet_n_is_directed_1_en_synsets/node")
    parser.add_argument('-e', "--epochs", type=int, default=1000, help='number of epochs')
    parser.add_argument('-b', "--batch_size", type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--hidden_states', help="list of hidden states sizes", nargs='+', type=lambda s: int(s),
                        default=(1024, 512))
    return parser.parse_args()
