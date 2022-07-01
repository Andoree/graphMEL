import codecs
import itertools
import os.path
import random
from argparse import ArgumentParser
from typing import List, Tuple, Set, Dict

import pandas as pd
from tqdm import tqdm

from graphmel.scripts.utils.io import read_mrrel


def get_concept_relation_triplet_strings(mrrel_df: pd.DataFrame) -> Set[str]:
    triplet_strings_set: Set[str] = set()

    for _, row in tqdm(mrrel_df.iterrows(), total=mrrel_df.shape[0], miniters=mrrel_df.shape[0] // 100):
        cui_1 = row["CUI1"]
        cui_2 = row["CUI2"]
        rel = row["REL"]
        triplet_string = f"{cui_1}||{rel}||{cui_2}"
        triplet_strings_set.add(triplet_string)
    return triplet_strings_set


def create_child_parent_adjacency_lists(triplet_strings_set: Set[str]) -> Dict[str, Set[str]]:
    adjacency_lists_dict: Dict[str, Set[str]] = {}

    for s in triplet_strings_set:
        cui_1, rel, cui_2 = s.split('||')
        if rel == "CHD" or rel == "RN":
            if adjacency_lists_dict.get(cui_2) is None:
                adjacency_lists_dict[cui_2] = set()
            adjacency_lists_dict[cui_2].add(cui_1)
        elif rel == "PAR" or rel == "RB":
            if adjacency_lists_dict.get(cui_1) is None:
                adjacency_lists_dict[cui_1] = set()
            adjacency_lists_dict[cui_1].add(cui_2)
    return adjacency_lists_dict


def count_childs(adjacency_lists_dict: Dict[str, Set[str]]) -> int:
    count = 0
    for cui_1, child_cuis_list in adjacency_lists_dict.items():
        count += len(child_cuis_list)
    return count


def count_interchild_parent_relations(adjacency_lists_dict: Dict[str, Set[str]]):
    num_relations = 0
    for cui_1, child_cuis_list in tqdm(adjacency_lists_dict.items(), total=len(adjacency_lists_dict.keys()),
                                       miniters=len(adjacency_lists_dict.keys()) // 100):
        num_siblings = len(child_cuis_list)
        if num_siblings > 1:
            assert len(list(itertools.combinations(child_cuis_list, 2))) == int(num_siblings * (num_siblings - 1) / 2)
            for (sibling_cui_1, sibling_cui_2) in itertools.combinations(child_cuis_list, 2):
                if adjacency_lists_dict.get(sibling_cui_1) is not None:
                    if sibling_cui_2 in adjacency_lists_dict[sibling_cui_1]:
                        print(f"cui_1: {cui_1}, sibling_cui_1: {sibling_cui_1}, sibling_cui_2: {sibling_cui_2}")
                        num_relations += 1
                if adjacency_lists_dict.get(sibling_cui_2) is not None:
                    if sibling_cui_1 in adjacency_lists_dict[sibling_cui_2]:
                        print(f"cui_1: {cui_1}, sibling_cui_2: {sibling_cui_2}, sibling_cui_1: {sibling_cui_1}")
                        num_relations += 1
                # assert not (sibling_cui_2 in adjacency_lists_dict[sibling_cui_1]
                #             and sibling_cui_1 in adjacency_lists_dict[sibling_cui_2])
    return num_relations


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrrel')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df_mrrel = read_mrrel(args.mrrel)
    triplet_strings_set = get_concept_relation_triplet_strings(mrrel_df=df_mrrel)
    examples = random.sample(triplet_strings_set, 5)
    for e in examples:
        print(e)
    adjacency_lists_dict = create_child_parent_adjacency_lists(triplet_strings_set=triplet_strings_set)
    num_childs = count_childs(adjacency_lists_dict=adjacency_lists_dict)
    interchild_parents = count_interchild_parent_relations(adjacency_lists_dict=adjacency_lists_dict)
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"Num triplets: {len(triplet_strings_set)}\nNum childs: {num_childs}\nInterchild parents: {interchild_parents}\n")


if __name__ == '__main__':
    main()
