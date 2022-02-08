import codecs
import os.path
from argparse import ArgumentParser
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd

from utils.read_umls import read_mrrel


def load_cui2node_ids_list(node2id_path: str) -> Dict[str, List[str]]:
    cui2node_ids_list_map = {}

    with codecs.open(node2id_path, 'r', encoding="utf-8", ) as inp_file:
        i = 0
        for line in inp_file:
            attrs = line.split('\t')
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
    for idx, row in mrrel_df.iterrows():
        cui_1 = row["CUI1"]
        cui_2 = row["CUI2"]
        cui_1_node_ids = cui2node_ids_list_map[cui_1]
        cui_2_node_ids = cui2node_ids_list_map[cui_2]
        for node_id_1, node_id_2 in product(cui_1_node_ids, cui_2_node_ids):
            edge_str = f"{node_id_1}~~~{node_id_2}"
            edges_str_set.add(edge_str)
            edge_str = f"{node_id_2}~~~{node_id_1}"
            edges_str_set.add(edge_str)
    edges_list = [(int(s.split('~~~')[0]), int(s.split('~~~')[1])) for s in edges_str_set]
    return edges_list


def main(args):
    mrrel_path = args.mrrel
    node2id_path = args.node2id_path
    output_link_path = args.output_link_path
    output_dir = os.path.dirname(output_link_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    mrrel_df = read_mrrel(mrrel_path)
    cui2node_ids_list_map = load_cui2node_ids_list(node2id_path)
    edges_list = generate_edges_list(mrrel_df=mrrel_df, cui2node_ids_list_map=cui2node_ids_list_map)
    with codecs.open(output_link_path, 'w+', encoding="utf-8") as out_file:
        for (node_id_1, node_id_2) in edges_list:
            out_file.write(f"{node_id_1}\t{node_id_2}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mrrel', type=str)
    parser.add_argument('--node2id_path', type=str)
    parser.add_argument('--output_link_path', type=str)
    args = parser.parse_args()
    main(args)
