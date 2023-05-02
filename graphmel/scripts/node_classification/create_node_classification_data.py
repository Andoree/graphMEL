import codecs
import logging
import os
from argparse import ArgumentParser
from typing import List, Dict, Set

import pandas as pd
from tqdm import tqdm

from graphmel.scripts.preprocessing.extract_tree_from_graph_dataset import create_hierarchy_adjacency_lists
from graphmel.scripts.utils.io import load_dict, load_node_id2terms_list, save_node_id2terms_list


def traverse_node(current_node_id: int,
                  visited_node_ids: Set[int],
                  parent_children_adj_lists: Dict[int, Set[int]],
                  node_depth: int,
                  node_id2sem_group: Dict[int, str],
                  res_n_id2class_label: Dict[int, List[str]]):
    curr_node_sem_gr = node_id2sem_group[current_node_id]
    class_label = f"{curr_node_sem_gr}-{node_depth}"
    if res_n_id2class_label.get(current_node_id) is None:
        res_n_id2class_label[current_node_id] = []
    res_n_id2class_label[current_node_id].append(class_label)

    for child_id in parent_children_adj_lists[current_node_id]:
        if child_id in visited_node_ids:
            continue
        visited_node_ids.add(child_id)
        traverse_node(current_node_id=child_id,
                      visited_node_ids=visited_node_ids,
                      parent_children_adj_lists=parent_children_adj_lists,
                      node_depth=node_depth + 1,
                      node_id2sem_group=node_id2sem_group,
                      res_n_id2class_label=res_n_id2class_label)
        visited_node_ids.remove(child_id)


def get_node_classes(parent_children_adj_lists: Dict[int, Set[int]],
                     child_parents_adj_lists: Dict[int, Set[int]],
                     node_id2sem_group: Dict[int, str],
                     node_id2terms_list: Dict[int, List[str]],
                     save_roots_dir: str) -> Dict[int, List[str]]:
    root_node_ids = [n_id for n_id, parent_ids in child_parents_adj_lists.items() if len(parent_ids) == 0]
    logging.info(f"There are {len(root_node_ids)} root nodes")
    save_roots_path = os.path.join(save_roots_dir, "root_concepts.txt")

    with codecs.open(save_roots_path, 'w+') as out_file:
        for n_id in root_node_ids:
            terms_str = '||'.join((t for t in node_id2terms_list[n_id]))
            out_file.write(f"{n_id}\t{terms_str}\n")
    res_n_id2class_label: Dict[int, List[str]] = {}

    visited_node_ids = set()
    node_depth = 0
    for node_id in tqdm(root_node_ids):
        visited_node_ids.add(node_id)
        traverse_node(current_node_id=node_id,
                      visited_node_ids=visited_node_ids,
                      parent_children_adj_lists=parent_children_adj_lists,
                      node_depth=node_depth + 1,
                      node_id2sem_group=node_id2sem_group,
                      res_n_id2class_label=res_n_id2class_label)
        visited_node_ids.remove(node_id)
    return res_n_id2class_label


def main(args):
    data_dir = args.input_data_dir
    edges_path = os.path.join(data_dir, "edges")
    rel2id_path = os.path.join(data_dir, "rel2id")
    node_id2sem_group_path = os.path.join(data_dir, "node_id2sem_group")
    node_id2terms_list_path = os.path.join(data_dir, "node_id2terms_list")
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)


    node_id2terms_list: Dict[int, List[str]] = load_node_id2terms_list(node_id2terms_list_path)

    rel2id = {rel: int(rel_id) for rel, rel_id in load_dict(rel2id_path).items()}
    id2rel = {v: k for k, v in rel2id.items()}
    keep_rel_ids = [rel2id[rel] for rel in args.keep_rels]
    edge_cols = ("src_n_id", "trg_n_id", "rel_id", "rela_id")
    logging.info("Loading edges")
    edges_df = pd.read_csv(edges_path, sep='\t', header=None,
                           names=edge_cols)

    logging.info(f"There are {edges_df.shape[0]} edges before filtering")
    edges_df = edges_df[edges_df["rel_id"].isin(keep_rel_ids)]
    logging.info(f"There are {edges_df.shape[0]} edges after REL filtering")
    it = (row["src_n_id"], row["trg_n_id"], row["rel_id"], row["rela_id"] for i, row in edges_df.iterrows())
    parent_childs_a_lst, child_parents_a_lst = create_hierarchy_adjacency_lists(edge_tuples=it, id2rel=id2rel)

    node_id2sem_group = pd.read_csv(node_id2sem_group_path, sep='\t', header=None,
                                    names=("node_id", "sem_group"))
    unique_sem_groups = node_id2sem_group["sem_group"].unique()
    logging.info(f"There are {len(unique_sem_groups)} sem groups: {unique_sem_groups}")

    res_n_id2class_label = get_node_classes(parent_children_adj_lists=parent_childs_a_lst,
                                            child_parents_adj_lists=child_parents_a_lst,
                                            node_id2sem_group=node_id2sem_group,
                                            node_id2terms_list=node_id2terms_list,
                                            save_roots_dir=args.save_tree_roots_dir)

    output_save_path = os.path.join(output_dir, "node_classes")
    save_node_id2terms_list(save_path=output_save_path, mapping=res_n_id2class_label,)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_data_dir')
    parser.add_argument('--sem_types_file')
    parser.add_argument('--keep_rels', nargs='+')
    parser.add_argument('--save_tree_roots_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
