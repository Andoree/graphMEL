import logging
import os
from argparse import ArgumentParser

import pandas as pd

from graphmel.scripts.utils.io import save_tuples, save_dict, save_node_id2terms_list
from graphmel.scripts.utils.io import read_mrconso, read_mrrel
from graphmel.scripts.utils.umls2graph import get_concept_list_groupby_cui, extract_umls_edges


def create_graph_files(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame, output_node_id2terms_list_path: str,
                       output_node_id2cui_path: str, output_edges_path: str, ignore_not_mapped_edges: bool):
    node_id2terms_list, node_id2cui = get_concept_list_groupby_cui(mrconso_df=mrconso_df)
    cui2node_id = {cui: node_id for node_id, cui in node_id2cui.items()}
    assert len(node_id2cui) == len(cui2node_id)
    logging.info("Generating edges....")
    edges = extract_umls_edges(mrrel_df, cui2node_id, ignore_not_mapped_edges=ignore_not_mapped_edges)
    logging.info("Saving the result....")
    save_node_id2terms_list(save_path=output_node_id2terms_list_path, mapping=node_id2terms_list, )
    save_dict(save_path=output_node_id2cui_path, dictionary=node_id2cui)
    save_tuples(save_path=output_edges_path, tuples=edges)


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrrel')
    parser.add_argument('--split_val', action="store_true")
    parser.add_argument('--train_proportion', type=float)
    parser.add_argument('--output_node_id2synonyms_path', )
    parser.add_argument('--output_node_id2cui_path', )
    parser.add_argument('--output_edges_path', )
    args = parser.parse_args()

    output_node_id2terms_list_path = args.output_node_id2synonyms_path
    output_node_id2cui_path = args.output_node_id2cui_path
    split_val = args.split_val
    output_edges_path = args.output_edges_path
    output_paths = (output_node_id2terms_list_path, output_node_id2cui_path, output_edges_path)
    for path in output_paths:
        output_dir = os.path.dirname(path)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)
    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)
    mrconso_df["CUI"] = mrconso_df["CUI"].fillna('')
    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(args.mrrel)

    logging.info("Generating node index....")
    if split_val:
        train_proportion = args.train_proportion
        num_rows = mrconso_df.shape[0]
        shuffled_mrconso = mrconso_df.sample(frac=1.0, random_state=42)
        del mrconso_df
        num_train_rows = int(num_rows * train_proportion)
        train_mrconso_df = shuffled_mrconso[:num_train_rows]
        val_mrconso_df = shuffled_mrconso[num_train_rows:]
        del shuffled_mrconso

        output_node_id2terms_list_dirname = os.path.dirname(output_node_id2terms_list_path)
        output_node_id2terms_list_filename = os.path.basename(output_node_id2terms_list_path)
        output_node_id2cui_dirname = os.path.dirname(output_node_id2cui_path)
        output_node_id2cui_filename = os.path.basename(output_node_id2cui_path)
        output_edges_dirname = os.path.dirname(output_edges_path)
        output_edges_filename = os.path.basename(output_edges_path)
        train_output_node_id2terms_list_path = os.path.join(output_node_id2terms_list_dirname,
                                                            f"train_{output_node_id2terms_list_filename}")
        val_output_node_id2terms_list_path = os.path.join(output_node_id2terms_list_dirname,
                                                          f"val_{output_node_id2terms_list_filename}")
        train_output_node_id2cui_path = os.path.join(output_node_id2cui_dirname, f"train_{output_node_id2cui_filename}")
        val_output_node_id2cui_path = os.path.join(output_node_id2cui_dirname, f"val_{output_node_id2cui_filename}")
        train_output_edges_path = os.path.join(output_edges_dirname, f"train_{output_edges_filename}")
        val_output_edges_path = os.path.join(output_edges_dirname, f"val_{output_edges_filename}")
        logging.info("Creating train graph files")
        create_graph_files(mrconso_df=train_mrconso_df, mrrel_df=mrrel_df,
                           output_node_id2terms_list_path=train_output_node_id2terms_list_path,
                           output_node_id2cui_path=train_output_node_id2cui_path,
                           output_edges_path=train_output_edges_path, ignore_not_mapped_edges=True)
        logging.info("Creating val graph files")
        create_graph_files(mrconso_df=val_mrconso_df, mrrel_df=mrrel_df,
                           output_node_id2terms_list_path=val_output_node_id2terms_list_path,
                           output_node_id2cui_path=val_output_node_id2cui_path,
                           output_edges_path=val_output_edges_path, ignore_not_mapped_edges=True)
    else:
        logging.info("Creating graph files")
        create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df,
                           output_node_id2terms_list_path=output_node_id2terms_list_path,
                           output_node_id2cui_path=output_node_id2cui_path,
                           output_edges_path=output_edges_path, ignore_not_mapped_edges=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main()
