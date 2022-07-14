import logging
import os.path
from argparse import ArgumentParser
import random
from typing import List, Tuple, Dict, Set

from graphmel.scripts.preprocessing.reformat_umls_to_graph import create_relations2id_dicts, create_cui2node_id_mapping, \
    create_graph_files
from graphmel.scripts.utils.io import write_strings, read_mrconso, read_mrrel, load_dict, load_edges_tuples, \
    save_adjacency_list
from tqdm import tqdm

HIERARCHICAL_RELS_LIST = ("CHD", "PAR", "RB", "RN")


def create_hierarchy_adjacency_lists(edge_tuples: List[Tuple[int]], id2rel: Dict[int, str]) -> \
        Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Given a list of graph edges, creates two hierarchical adjacency
    lists based on hierarcical types of "REL" attributes
    :param edge_tuples: List of edge tuples. Each tuple contains source and
    target nodes and relation types ("REL" and "RELA" attributes)
    :param id2rel: inverse mapping for "REL" -> "REL's id"
    :return: Two adjacency list structures.
    The first one contains childs grouped by their parent. The second one returns all parents of a given child.
    """
    parent_childs_adjacency_list: Dict[int, Set[int]] = {}
    child_parents_adjacency_list: Dict[int, Set[int]] = {}
    num_removed_self_loops = 0
    logging.info(f"Processing edge tuples. There are {len(edge_tuples)} tuples.")
    for t in tqdm(edge_tuples, miniters=len(edge_tuples) // 100):
        node_id_1 = t[0]
        node_id_2 = t[1]
        if node_id_1 != node_id_2:
            rel_id = t[2]
            rel_verbose = id2rel[rel_id]
            if rel_verbose in ("CHD", "RN"):
                parent_node_id, child_node_id = node_id_2, node_id_1
            elif rel_verbose in ("PAR", "RB"):
                parent_node_id, child_node_id = node_id_1, node_id_2
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

    input_edges_path = os.path.join(input_graph_dataset_dir, "edges")
    input_rel2rel_id_path = os.path.join(input_graph_dataset_dir, f"rel2id")

    output_parent_childs_adjacency_list_path = os.path.join(output_dir, "parent_childs_adjacency_list")
    output_child_parents_adjacency_list_path = os.path.join(output_dir, "child_parents_adjacency_list")

    rel2id = {rel: int(i) for rel, i in load_dict(input_rel2rel_id_path)}
    id2rel = {i: rel for rel, i in rel2id.items()}
    edge_tuples = load_edges_tuples(path=input_edges_path)
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
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()
    main(arguments)
