import logging
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def get_concept_list_groupby_cui(mrconso_df: pd.DataFrame) -> (Dict[int, List[str]], Dict[int, str]):
    logging.info("Started creating CUI to terms mapping")
    node_id2terms_list: Dict[int, List[str]] = {}
    node_id2cui: Dict[int, str] = {}
    for node_id, row in tqdm(mrconso_df.iterrows(), miniters=mrconso_df.shape[0] // 50):
        cui = row["CUI"].strip()
        term_str = row["STR"].strip()
        if term_str == '':
            continue
        node_id2cui[node_id] = cui
        if node_id2terms_list.get(node_id) is None:
            node_id2terms_list[node_id] = []
        node_id2terms_list[node_id].append(term_str.strip())
    logging.info("CUI to terms mapping is created")
    return node_id2terms_list, node_id2cui


def extract_umls_edges(mrrel_df: pd.DataFrame, cui2node_id: Dict[str, int], ignore_not_mapped_edges=False) -> List[
    Tuple[int, int]]:
    cui_str_set = set()
    logging.info("Started generating graph edges")
    edges: List[Tuple[int, int]] = []
    not_mapped_edges_counter = 0
    for idx, row in tqdm(mrrel_df.iterrows(), miniters=mrrel_df.shape[0] // 100, total=mrrel_df.shape[0]):
        cui_1 = row["CUI1"].strip()
        cui_2 = row["CUI2"].strip()
        if cui_1 > cui_2:
            cui_1, cui_2 = cui_2, cui_1
        if cui2node_id.get(cui_1) is not None and cui2node_id.get(cui_2) is not None:
            two_cuis_str = f"{cui_1}~~{cui_2}"
            if two_cuis_str not in cui_str_set:
                cui_1_node_id = cui2node_id[cui_1]
                cui_2_node_id = cui2node_id[cui_2]
                edges.append((cui_1_node_id, cui_2_node_id))
                edges.append((cui_2_node_id, cui_1_node_id))
            cui_str_set.add(two_cuis_str)
        else:
            if not ignore_not_mapped_edges:
                raise AssertionError(f"Either CUI {cui_1} or {cui_2} are not found in CUI2node_is mapping")
            else:
                not_mapped_edges_counter += 1
    if not ignore_not_mapped_edges:
        logging.info(f"{not_mapped_edges_counter} edges are not mapped to any node")
    logging.info(f"Finished generating edges."
                 f"There are {len(edges)} non-oriented edges")

    return edges
