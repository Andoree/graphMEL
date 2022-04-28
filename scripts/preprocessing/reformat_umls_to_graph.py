import logging
import os
from argparse import ArgumentParser
from typing import Dict

import pandas as pd

from graphmel.scripts.utils.io import save_tuples, save_dict, save_node_id2terms_list
from graphmel.scripts.utils.io import read_mrconso, read_mrrel
from graphmel.scripts.utils.umls2graph import get_concept_list_groupby_cui, extract_umls_oriented_edges_with_relations


def create_graph_files(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame, rel2id: Dict[str, int],
                       rela2id: Dict[str, int], output_node_id2terms_list_path: str,
                       output_node_id2cui_path: str, output_edges_path: str, output_rel2rel_id_path: str,
                       output_rela2rela_id_path, ignore_not_mapped_edges: bool):
    node_id2terms_list, node_id2cui, cui2node_id = get_concept_list_groupby_cui(mrconso_df=mrconso_df)
    logging.info("Generating edges....")

    edges = extract_umls_oriented_edges_with_relations(mrrel_df=mrrel_df, cui2node_id=cui2node_id,
                                                       rel2rel_id=rel2id, rela2rela_id=rela2id,
                                                       ignore_not_mapped_edges=ignore_not_mapped_edges)

    logging.info("Saving the result....")
    save_node_id2terms_list(save_path=output_node_id2terms_list_path, mapping=node_id2terms_list, )
    save_dict(save_path=output_node_id2cui_path, dictionary=node_id2cui)
    save_dict(save_path=output_rel2rel_id_path, dictionary=rel2id)
    save_dict(save_path=output_rela2rela_id_path, dictionary=rela2id)
    save_tuples(save_path=output_edges_path, tuples=edges)


def create_relations2id_dicts(mrrel_df: pd.DataFrame):
    mrrel_df.REL.fillna("NAN", inplace=True)
    mrrel_df.RELA.fillna("NAN", inplace=True)
    rel2id = {rel: rel_id for rel_id, rel in enumerate(mrrel_df.REL.unique())}
    rela2id = {rela: rela_id for rela_id, rela in enumerate(mrrel_df.RELA.unique())}
    rel2id["LOOP"] = max(rel2id.values()) + 1
    rela2id["LOOP"] = max(rela2id.values()) + 1
    logging.info(f"There are {len(rel2id.keys())} unique RELs and {len(rela2id.keys())} unique RELAs")
    print("REL2REL_ID", rel2id)
    print("RELA2RELA_ID", rela2id)
    return rel2id, rela2id


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrrel')
    parser.add_argument('--split_val', action="store_true")
    parser.add_argument('--train_proportion', type=float)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    split_val = args.split_val
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)
    mrconso_df["STR"].fillna('', inplace=True)
    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(args.mrrel)

    logging.info("Generating node index....")
    rel2id, rela2id = create_relations2id_dicts(mrrel_df)
    if split_val:
        train_dir = os.path.join(output_dir, "train/")
        val_dir = os.path.join(output_dir, "val/")
        for d in (train_dir, val_dir):
            if not os.path.exists(d):
                os.makedirs(d)
        train_proportion = args.train_proportion
        num_rows = mrconso_df.shape[0]
        shuffled_mrconso = mrconso_df.sample(frac=1.0, random_state=42)
        del mrconso_df
        num_train_rows = int(num_rows * train_proportion)
        train_mrconso_df = shuffled_mrconso[:num_train_rows]
        val_mrconso_df = shuffled_mrconso[num_train_rows:]
        del shuffled_mrconso

        train_output_node_id2terms_list_path = os.path.join(train_dir, "node_id2terms_list")
        val_output_node_id2terms_list_path = os.path.join(val_dir, "node_id2terms_list")
        train_output_node_id2cui_path = os.path.join(train_dir, "id2cui")
        val_output_node_id2cui_path = os.path.join(val_dir, "id2cui")
        train_output_edges_path = os.path.join(train_dir, "edges")
        val_output_edges_path = os.path.join(val_dir, "edges")

        train_output_rel2rel_id_path = os.path.join(train_dir, "rel2id")
        val_output_rel2rel_id_path = os.path.join(val_dir, "rel2id")
        train_output_rela2rela_id_path = os.path.join(train_dir, "rela2id")
        val_output_rela2rela_id_path = os.path.join(val_dir, "rela2rid")

        logging.info("Creating train graph files")
        create_graph_files(mrconso_df=train_mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                           output_node_id2terms_list_path=train_output_node_id2terms_list_path,
                           output_node_id2cui_path=train_output_node_id2cui_path,
                           output_edges_path=train_output_edges_path,
                           output_rel2rel_id_path=train_output_rel2rel_id_path,
                           output_rela2rela_id_path=train_output_rela2rela_id_path, ignore_not_mapped_edges=True, )
        logging.info("Creating val graph files")
        create_graph_files(mrconso_df=val_mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                           output_node_id2terms_list_path=val_output_node_id2terms_list_path,
                           output_node_id2cui_path=val_output_node_id2cui_path,
                           output_edges_path=val_output_edges_path, output_rel2rel_id_path=val_output_rel2rel_id_path,
                           output_rela2rela_id_path=val_output_rela2rela_id_path,
                           ignore_not_mapped_edges=True, )
    else:
        logging.info("Creating graph files")
        output_node_id2terms_list_path = os.path.join(output_dir, "node_id2terms_list")
        output_node_id2cui_path = os.path.join(output_dir, "id2cui")
        output_edges_path = os.path.join(output_dir, "edges")
        output_rel2rel_id_path = os.path.join(output_dir, f"rel2id")
        output_rela2rela_id_path = os.path.join(output_dir, f"rela2id")
        create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                           output_node_id2terms_list_path=output_node_id2terms_list_path,
                           output_node_id2cui_path=output_node_id2cui_path,
                           output_edges_path=output_edges_path, output_rel2rel_id_path=output_rel2rel_id_path,
                           output_rela2rela_id_path=output_rela2rela_id_path, ignore_not_mapped_edges=True, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main()
