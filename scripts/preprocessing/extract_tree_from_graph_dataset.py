# TODO


import logging
import os.path
from argparse import ArgumentParser

import random

from graphmel.scripts.preprocessing.reformat_umls_to_graph import create_relations2id_dicts, create_cui2node_id_mapping, \
    create_graph_files
from graphmel.scripts.utils.io import write_strings, read_mrconso, read_mrrel, load_dict


def main(args):
    random.seed(42)
    output_dir = args.output_dir
    input_graph_dataset_dir = args.input_graph_dataset_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)


    # logging.info("Loading MRCONSO....")
    # mrconso_df = read_mrconso(args.mrconso)
    # logging.info(f"MRCONSO is loaded. There are {mrconso_df.shape[0]} rows")
    #
    # if args.ontology is not None:
    #     mrconso_df = mrconso_df[mrconso_df.SAB.isin(args.ontology)]
    #     last_subdir = '_'.join(args.ontology) + f"_{last_subdir}"
    # logging.info(f"There are {mrconso_df.shape[0]} MRCONSO rows after ontology filtering")
    # if args.langs is not None:
    #     mrconso_df = mrconso_df[mrconso_df.LAT.isin(args.langs)]
    #     last_subdir = '_'.join(args.langs) + f"_{last_subdir}"
    # output_dir = os.path.join(output_dir, last_subdir)
    # if not os.path.exists(output_dir) and output_dir != '':
    #     os.makedirs(output_dir)

    # logging.info(f"There are {mrconso_df.shape[0]} MRCONSO rows after language filtering")
    # mrconso_df["STR"].fillna('', inplace=True)

    # logging.info("Loading MRREL....")
    # mrrel_df = read_mrrel(args.mrrel)
    # rel2id, rela2id = create_relations2id_dicts(mrrel_df)
    # cui2node_id = create_cui2node_id_mapping(mrconso_df=mrconso_df)

    # logging.info("Creating graph files")
    # TODO input_graph_dataset_dir
    input_node_id2terms_list_path = os.path.join(input_graph_dataset_dir, "node_id2terms_list")
    input_node_id2cui_path = os.path.join(input_graph_dataset_dir, "id2cui")
    input_edges_path = os.path.join(input_graph_dataset_dir, "edges")
    input_rel2rel_id_path = os.path.join(input_graph_dataset_dir, f"rel2id")
    input_rela2rela_id_path = os.path.join(input_graph_dataset_dir, f"rela2id")


    create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                       cui2node_id=cui2node_id,
                       output_node_id2terms_list_path=output_node_id2terms_list_path,
                       output_node_id2cui_path=output_node_id2cui_path,
                       output_edges_path=output_edges_path, output_rel2rel_id_path=output_rel2rel_id_path,
                       output_rela2rela_id_path=output_rela2rela_id_path, ignore_not_mapped_edges=True, )
    pos_pairs = generate_positive_pairs(mrconso_df=mrconso_df, mrrel_df=mrrel_df, cui2node_id=cui2node_id)
    if args.split_val:
        output_train_pos_pairs_path = os.path.join(output_dir, f"train_pos_pairs")
        output_val_pos_pairs_path = os.path.join(output_dir, f"val_pos_pairs")

        train_proportion = args.train_proportion
        num_pos_pairs = len(pos_pairs)
        # Shuffling positive pairs
        random.shuffle(pos_pairs)
        num_train_pos_pairs = int(num_pos_pairs * train_proportion)
        train_pos_pairs = pos_pairs[:num_train_pos_pairs]
        val_pos_pairs = pos_pairs[num_train_pos_pairs:]
        logging.info(f"Positive pairs are split: {len(train_pos_pairs)} in train and {len(val_pos_pairs)} in val")
        logging.info(f"Train positive pair examples:\n" + '\n'.join(train_pos_pairs[:3]))
        logging.info(f"Val positive pair examples:\n" + '\n'.join(val_pos_pairs[:3]))
        write_strings(fpath=output_train_pos_pairs_path, strings_list=train_pos_pairs)
        write_strings(fpath=output_val_pos_pairs_path, strings_list=val_pos_pairs)
    else:
        random.shuffle(pos_pairs)
        output_pos_pairs_path = os.path.join(output_dir, f"train_pos_pairs")
        logging.info(f"Positive pair examples:\n" + '\n'.join(pos_pairs[:3]))
        write_strings(fpath=output_pos_pairs_path, strings_list=pos_pairs)

# TODO: Посмотреть что в мтеодах, а потом их полностью переделать
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    # parser.add_argument('--mrconso', type=str)
    # parser.add_argument('--mrrel', type=str)
    # parser.add_argument('--langs', nargs='+', default=None)
    # parser.add_argument('--split_val', action="store_true")
    # parser.add_argument('--train_proportion', type=float)
    # parser.add_argument('--ontology', default=None, nargs='+')
    parser.add_argument('--input_graph_dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()

    main(arguments)

