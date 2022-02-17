import codecs
import gc
import os.path
from argparse import ArgumentParser
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd

from utils.io import read_mrrel
from tqdm import tqdm
import logging


def load_cui2node_ids_list(node2id_path: str) -> Dict[str, List[str]]:
    cui2node_ids_list_map = {}

    with codecs.open(node2id_path, 'r', encoding="utf-8", ) as inp_file:
        i = 0
        for line in inp_file:
            attrs = line.strip().split('\t')
            node_id = attrs[0]
            concept_str = attrs[1]
            cui = attrs[2]
            if cui2node_ids_list_map.get(cui) is None:
                cui2node_ids_list_map[cui] = []
            cui2node_ids_list_map[cui].append(node_id)
            i += 1
    # Number of concepts validation
    s = 0
    for lst in cui2node_ids_list_map.values():
        s += len(lst)
    assert i == s
    return cui2node_ids_list_map


def generate_edges_list(mrrel_df: pd.DataFrame, cui2node_ids_list_map: Dict[str, List[str]]) -> List[Tuple[int, int]]:
    edges_str_set = set()
    cui_str_set = set()
    not_matched_cui_count = 0
    for idx, row in tqdm(mrrel_df.iterrows(), miniters=mrrel_df.shape[0] // 500, total=mrrel_df.shape[0]):
        cui_1 = row["CUI1"]
        cui_2 = row["CUI2"]
        if cui_1 > cui_2:
            cui_1, cui_2 = cui_2, cui_1
        if cui2node_ids_list_map.get(cui_1) is not None and cui2node_ids_list_map.get(cui_2) is not None:
            two_cuis_str = f"{cui_1}~~{cui_2}"
            # cui_2_cui_1_str = f"{cui_2}~~~{cui_1}"
            if two_cuis_str not in cui_str_set:
                cui_1_node_ids = cui2node_ids_list_map[cui_1]
                cui_2_node_ids = cui2node_ids_list_map[cui_2]
                for node_id_1, node_id_2 in product(cui_1_node_ids, cui_2_node_ids):
                    if node_id_1 > node_id_2:
                        node_id_1, node_id_2 = node_id_2, node_id_1
                    edge_str = f"{node_id_1}~~{node_id_2}"
                    edges_str_set.add(edge_str)
                    # edge_str = f"{node_id_2}~~~{node_id_1}"
                    # edges_str_set.add(edge_str)
            cui_str_set.add(two_cuis_str)
            # cui_str_set.add(cui_2_cui_1_str)
        else:
            not_matched_cui_count += 1
    logging.info(f"Finished generating edges."
                 f"There are {len(edges_str_set)} non-oriented edges"
                 f"{not_matched_cui_count} CUIs are not matched to any node\n"
                 f"Converting edges set to list....")

    edges_list = [(int(s.split('~~')[0]), int(s.split('~~')[1])) for s in edges_str_set]
    logging.info("Finished converting edges set to list")
    del edges_str_set
    logging.info(f"Removing temporary files. Freed space: {gc.collect()}")
    return edges_list


def main(args):
    mrrel_path = args.mrrel
    node2id_path = args.node2id_path
    output_link_path = args.output_link_path
    output_dir = os.path.dirname(output_link_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(mrrel_path)
    logging.info("Loading CUI to node id map....")
    cui2node_ids_list_map = load_cui2node_ids_list(node2id_path)
    logging.info("Creating graph edges....")
    edges_list = generate_edges_list(mrrel_df=mrrel_df, cui2node_ids_list_map=cui2node_ids_list_map)
    logging.info("Saving graph edges....")
    with codecs.open(output_link_path, 'w+', encoding="utf-8") as out_file:
        buffer = []
        for (node_id_1, node_id_2) in tqdm(edges_list, miniters=len(edges_list) // 500):
            buffer.append((node_id_1, node_id_2))
            if len(buffer) > 100000:
                s = "".join((f"{t[0]}\t{t[1]}\n{t[1]}\t{t[0]}\n" for t in buffer))
                out_file.write(s)
                buffer.clear()
        if len(buffer) > 0:
            s = "".join((f"{t[0]}\t{t[1]}\n{t[1]}\t{t[0]}\n" for t in buffer))
            out_file.write(s)

            # out_file.write(f"{node_id_1}\t{node_id_2}\n")
            # out_file.write(f"{node_id_2}\t{node_id_1}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrrel', type=str)
    parser.add_argument('--node2id_path', type=str)
    parser.add_argument('--output_link_path', type=str)
    args = parser.parse_args()
    main(args)
