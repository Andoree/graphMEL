import codecs
import logging
import os.path
from argparse import ArgumentParser
import random
from graphmel.scripts.utils.io import load_dict, read_mrsty, read_sem_groups, save_dict
from typing import Dict

import pandas as pd


def map_node_id2sem_group(cui2node_id: Dict[str, int], tui2sem_group: Dict[str, str], mrsty_df: pd.DataFrame) \
        -> Dict[int, str]:
    node_id2sem_group = {}
    for _, row in mrsty_df.iterrows():

        cui = row["CUI"].strip()
        tui = row["TUI"].strip()
        if cui2node_id.get(cui) is not None:
            semantic_group_of_tui = tui2sem_group[tui]
            node_id = cui2node_id[cui]
            node_id2sem_group[node_id] = semantic_group_of_tui
    return node_id2sem_group


def main(args):
    random.seed(42)

    graph_dataset_dir = args.graph_dataset_dir

    mrsty_df = read_mrsty(fpath=args.mrsty)
    input_id2cui_id_path = os.path.join(graph_dataset_dir, f"id2cui")
    output_node_id2sem_group_path = os.path.join(graph_dataset_dir, f"node_id2sem_group")

    sem_groups_df = read_sem_groups(fpath=args.sem_groups_file)
    tui2sem_group = {row["TUI"]: row["Semantic Group Abbrev"] for _, row in sem_groups_df.iterrows()}
    node_id2cui = {int(i): cui for i, cui in load_dict(input_id2cui_id_path).items()}
    cui2node_id = {cui: i for i, cui in node_id2cui.items()}

    node_id2sem_group = map_node_id2sem_group(cui2node_id=cui2node_id, tui2sem_group=tui2sem_group, mrsty_df=mrsty_df)

    save_dict(save_path=output_node_id2sem_group_path, dictionary=node_id2sem_group, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--graph_dataset_dir', type=str)
    parser.add_argument('--sem_groups_file', type=str)
    parser.add_argument('--mrsty', type=str)

    arguments = parser.parse_args()
    main(arguments)
