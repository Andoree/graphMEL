import codecs
import logging
import os.path
from argparse import ArgumentParser
import random
from typing import List, Tuple, Dict, Set

import pandas as pd

from graphmel.scripts.preprocessing.reformat_umls_to_graph import create_relations2id_dicts, create_cui2node_id_mapping, \
    create_graph_files
from graphmel.scripts.utils.io import write_strings, read_mrconso, read_mrrel, load_dict, load_edges_tuples, \
    save_adjacency_list, read_mrsty
from tqdm import tqdm


def create_mrsty_index(mrsty_df: pd.DataFrame, initial_index: int) -> Tuple[Dict[str, int], Dict[str, str]]:
    unique_tuis = set()
    tui2verbose: Dict[str, str] = {}
    for _, row in mrsty_df.iterrows():
        tui = row["TUI"].strip()
        sem_type_verbose = row["STY"].strip()

        unique_tuis.add(tui)
        if tui2verbose.get(tui) is None:
            tui2verbose[tui] = sem_type_verbose
    tui2node_id = {tui: initial_index + i for i, tui in enumerate(sorted(unique_tuis))}
    return tui2node_id, tui2verbose


def write_semantic_type_nodes_with_names(tui2node_id: Dict[str, int], tui2verbose: Dict[str, str],
                                         node2terms_path: str, node_id_sep: str = '\t', ):
    with codecs.open(node2terms_path, 'a', encoding="utf-8") as out_file:
        for tui in tui2node_id.keys():
            node_id = tui2node_id[tui]
            tui_verbose = tui2verbose[tui]
            out_file.write(f"{node_id}{node_id_sep}{tui_verbose}\n")


def add_semantic_type_hierarchical_edge_tuples(mrsty_df: pd.DataFrame, edge_tuples: List[Tuple[int, int, int, int]], rel2id: Dict[str, int],
                                               cui2id: Dict[str, int], tui2node_id: Dict[str, int]):
    logging.info("Adding additional MRSTY nodes")
    for _, row in tqdm(mrsty_df.iterrows(), total=mrsty_df.shape[0], miniters=mrsty_df.shape[0] // 100):
        cui = row["CUI"]
        tui = row["TUI"]

        cui_node_id = cui2id[cui]
        tui_node_id = tui2node_id[tui]
        parent_rel_id = rel2id["PAR"]
        child_rel_id = rel2id["CHD"]
        edge_tuples.append((cui_node_id, tui_node_id, parent_rel_id, 0))
        edge_tuples.append((tui_node_id, cui_node_id, child_rel_id, 0))
    logging.info("Finished adding additional MRSTY nodes")


def find_max_node_id(edge_tuples: List[Tuple[int, int, int, int]]) -> int:
    max_node_id = -1
    for t in edge_tuples:
        node_id_1 = t[0]
        node_id_2 = t[1]
        max_node_id = max(max_node_id, node_id_1, node_id_2)
    return max_node_id


def create_hierarchy_adjacency_lists(edge_tuples: List[Tuple[int]], id2rel: Dict[int, str],) \
        -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Given a list of graph edges, creates two hierarchical adjacency
    lists based on hierarcical types of "REL" attributes
    :param edge_tuples: List of edge tuples. Each tuple contains source and
    target nodes and relation types ("REL" and "RELA" attributes)
    :param id2rel: inverse mapping for "REL" -> "REL's id"
    :param mrsty_df: MRSTY.RRF dataframe
    :return: Two adjacency list structures.
    The first one contains childs grouped by their parent. The second one returns all parents of a given child.
    """
    parent_childs_adjacency_list: Dict[int, Set[int]] = {}
    child_parents_adjacency_list: Dict[int, Set[int]] = {}
    num_removed_self_loops = 0
    logging.info(f"Processing edge tuples. There are {len(edge_tuples)} tuples.")
    unique_node_ids = set()
    for t in tqdm(edge_tuples, miniters=len(edge_tuples) // 100):
        node_id_1 = t[0]
        node_id_2 = t[1]
        unique_node_ids.add(node_id_1)
        unique_node_ids.add(node_id_2)
        if node_id_1 != node_id_2:
            rel_id = t[2]
            rel_verbose = id2rel[rel_id]
            if rel_verbose in ("CHD", "RN"):
                parent_node_id, child_node_id = node_id_1, node_id_2
            elif rel_verbose in ("PAR", "RB"):
                parent_node_id, child_node_id = node_id_2, node_id_1
            else:
                continue
            if parent_childs_adjacency_list.get(parent_node_id) is None:
                parent_childs_adjacency_list[parent_node_id] = set()
            if child_parents_adjacency_list.get(child_node_id) is None:
                child_parents_adjacency_list[child_node_id] = set()
            parent_childs_adjacency_list[parent_node_id].add(child_node_id)
            child_parents_adjacency_list[child_node_id].add(parent_node_id)
        else:
            num_removed_self_loops += 1
    num_tree_edges = 0
    for child_ids in parent_childs_adjacency_list.values():
        num_tree_edges += len(child_ids)
    for parent_ids in child_parents_adjacency_list.values():
        num_tree_edges += len(parent_ids)
    logging.info(f"Finished extracting hierarchy tree. There are {num_tree_edges} edges in the tree. "
                 f"Removed {num_removed_self_loops} self-loops")
    return parent_childs_adjacency_list, child_parents_adjacency_list


def main(args):
    random.seed(42)
    output_dir = args.output_dir
    input_graph_dataset_dir = args.input_graph_dataset_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    mrsty_df = read_mrsty(fpath=args.mrsty)
    input_edges_path = os.path.join(input_graph_dataset_dir, "edges")
    input_rel2rel_id_path = os.path.join(input_graph_dataset_dir, f"rel2id")
    input_id2cui_id_path = os.path.join(input_graph_dataset_dir, f"id2cui")
    input_node_id2terms_list_path = os.path.join(input_graph_dataset_dir, f"node_id2terms_list")

    output_parent_childs_adjacency_list_path = os.path.join(output_dir, "parent_childs_adjacency_list")
    output_child_parents_adjacency_list_path = os.path.join(output_dir, "child_parents_adjacency_list")

    id2cui = {int(i): cui for i, cui in load_dict(input_id2cui_id_path)}
    cui2id = {cui: i for i, cui in id2cui.items()}
    rel2id = {rel: int(i) for rel, i in load_dict(input_rel2rel_id_path)}
    id2rel = {i: rel for rel, i in rel2id.items()}
    edge_tuples = load_edges_tuples(path=input_edges_path)
    
    num_concept_nodes = find_max_node_id(edge_tuples=edge_tuples)
    tui2node_id, tui2verbose = create_mrsty_index(mrsty_df=mrsty_df, initial_index=num_concept_nodes)

    add_semantic_type_hierarchical_edge_tuples(mrsty_df=mrsty_df, edge_tuples=edge_tuples, rel2id=rel2id, cui2id=cui2id,
                                               tui2node_id=tui2node_id)
    write_semantic_type_nodes_with_names(tui2node_id=tui2node_id, tui2verbose=tui2verbose,
                                         node2terms_path=input_node_id2terms_list_path,)

    parent_childs_adjacency_list, child_parents_adjacency_list = \
        create_hierarchy_adjacency_lists(edge_tuples=edge_tuples, id2rel=id2rel)
    save_adjacency_list(adjacency_list=parent_childs_adjacency_list,
                        save_path=output_parent_childs_adjacency_list_path, )
    save_adjacency_list(adjacency_list=child_parents_adjacency_list,
                        save_path=output_child_parents_adjacency_list_path, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_graph_dataset_dir', type=str)
    parser.add_argument('--mrsty', type=str)
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()
    main(arguments)
