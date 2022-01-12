import codecs
import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple

from ..utils import read_mrrel, read_mrconso
import pandas as pd


def create_edges_list(mrrel_df: pd.Dataframe, CUI_to_node_id_dict: Dict[str, int]) \
        -> List[Tuple[int, int]]:
    edges = []
    for idx, row in mrrel_df.iterrows():
        cui_1 = row["CUI1"]
        cui_2 = row["CUI2"]
        node_id_1 = CUI_to_node_id_dict[cui_1]
        node_id_2 = CUI_to_node_id_dict[cui_2]
        edges.append((node_id_1, node_id_2))
    return edges


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrrel')
    parser.add_argument('--output_link_path', )
    parser.add_argument('--output_node_path', )
    parser.add_argument('--output_node_index_path', )
    args = parser.parse_args()

    output_link_path = args.output_link_path
    output_node_path = args.output_node_path
    output_node_index_path = args.output_node_index_path
    output_dir = os.path.dirname(output_node_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_dir = os.path.dirname(output_link_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_dir = os.path.dirname(output_node_index_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    mrrel_df = read_mrrel(args.mrrel)
    mrconso_df = read_mrconso(args.mrconso)

    unique_CUIs_list = set(mrconso_df["CUI"].unique())
    CUI_to_node_id_dict = {cui: idx for idx, cui in enumerate(unique_CUIs_list)}
    edges_list = create_edges_list(mrrel_df=mrrel_df, CUI_to_node_id_dict=CUI_to_node_id_dict)
    with codecs.open(output_node_path, 'w+', encoding="utf-8") as node_file, \
            codecs.open(output_link_path, 'w+', encoding="utf-8") as link_file, \
            codecs.open(output_node_index_path, 'w+', encoding="utf-8") as node_index_file:
        for cui, node_id in CUI_to_node_id_dict.items():
            node_index_file.write(f"{cui}\t{node_id}\n")
        for (node_id_1, node_id_2) in edges_list:
            link_file.write(f"{node_id_1}\t{node_id_2}\n")
        for cui, node_id in CUI_to_node_id_dict.items():
            node_file.write(f"{node_id}\t0\t0\n")


if __name__ == '__main__':
    main()
