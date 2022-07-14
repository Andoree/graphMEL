import codecs
import itertools
import os.path
import random
from argparse import ArgumentParser
from typing import List, Tuple, Set, Dict
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm

from graphmel.scripts.utils.io import read_mrrel, read_mrconso


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


def write_intersiblings_hierarhical_relations(adjacency_lists_dict: Dict[str, Set[str]], cui2term: Dict[str, str],
                                              output_path: str):
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"common_parent||parent_sibling||child_sibling\n")
        for superparent_cui, child_cuis_list in tqdm(adjacency_lists_dict.items(), total=len(adjacency_lists_dict.keys()),
                                           miniters=len(adjacency_lists_dict.keys()) // 100):
            num_siblings = len(child_cuis_list)
            if num_siblings > 1:
                assert len(list(itertools.combinations(child_cuis_list, 2))) == int(
                    num_siblings * (num_siblings - 1) / 2)
                for (sibling_cui_1, sibling_cui_2) in itertools.combinations(child_cuis_list, 2):
                    parent_sibling_cui, child_sibling_cui = None, None
                    if adjacency_lists_dict.get(sibling_cui_1) is not None:
                        if sibling_cui_2 in adjacency_lists_dict[sibling_cui_1]:
                            parent_sibling_cui, child_sibling_cui = sibling_cui_1, sibling_cui_2
                    if adjacency_lists_dict.get(sibling_cui_2) is not None:
                        if sibling_cui_1 in adjacency_lists_dict[sibling_cui_2]:
                            parent_sibling_cui, child_sibling_cui = sibling_cui_2, sibling_cui_1
                    if parent_sibling_cui is not None and child_sibling_cui is not None \
                            and cui2term.get(parent_sibling_cui) is not None and cui2term.get(
                        child_sibling_cui) is not None and cui2term.get(superparent_cui) is not None:
                        parent_sibling_str = cui2term[parent_sibling_cui]
                        child_sibling_str =  cui2term[child_sibling_cui]
                        superparent_str = cui2term[superparent_cui]
                        out_file.write(f"{superparent_cui},{superparent_str}||{parent_sibling_cui},"
                                       f"{parent_sibling_str}||{child_sibling_cui},{child_sibling_str}\n")


def create_cui2term_dict(mrconso_df: pd.DataFrame) -> Dict[str, str]:
    cui2term = {}
    for _, row in tqdm(mrconso_df.iterrows(), total=mrconso_df.shape[0], miniters=mrconso_df.shape[0] // 100):
        if row["CUI"] is not np.nan and row["STR"] is not np.nan:
            cui = row["CUI"].strip()
            term = row["STR"].strip()
            cui2term[cui] = term
        else:
            logging.info(f"nan MRCONSO str, cui {cui}, str {term}")
    return cui2term


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrrel')
    parser.add_argument('--mrconso')
    parser.add_argument('--ru_output_path')
    parser.add_argument('--en_output_path')
    args = parser.parse_args()

    ru_output_path = args.ru_output_path
    output_dir = os.path.dirname(ru_output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    df_mrconso = read_mrconso(args.mrconso)
    ru_df_mrconso = df_mrconso[df_mrconso["LAT"] == "RUS"]
    en_df_mrconso = df_mrconso[df_mrconso["LAT"] == "ENG"]
    del df_mrconso
    ru_cui2term = create_cui2term_dict(mrconso_df=ru_df_mrconso)
    en_cui2term = create_cui2term_dict(mrconso_df=en_df_mrconso)
    del ru_df_mrconso
    del en_df_mrconso
    df_mrrel = read_mrrel(args.mrrel)
    triplet_strings_set = get_concept_relation_triplet_strings(mrrel_df=df_mrrel)
    examples = random.sample(triplet_strings_set, 5)
    for e in examples:
        print(e)
    adjacency_lists_dict = create_child_parent_adjacency_lists(triplet_strings_set=triplet_strings_set)
    write_intersiblings_hierarhical_relations(adjacency_lists_dict=adjacency_lists_dict, cui2term=ru_cui2term,
                                              output_path=args.ru_output_path)
    write_intersiblings_hierarhical_relations(adjacency_lists_dict=adjacency_lists_dict, cui2term=en_cui2term,
                                              output_path=args.en_output_path)


if __name__ == '__main__':
    main()
