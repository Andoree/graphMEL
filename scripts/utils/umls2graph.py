import logging
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def get_concept_list_groupby_cui(mrconso_df: pd.DataFrame) -> (Dict[int, List[str]], Dict[int, str], Dict[str, int]):
    logging.info("Started creating CUI to terms mapping")
    node_id2terms_list: Dict[int, List[str]] = {}
    # TODO: Always remember that I delete duplicated pairs here.
    logging.info(f"Removing duplicated (CUI, STR) pairs, {mrconso_df.shape[0]} rows before deletion")
    mrconso_df.drop_duplicates(subset=("CUI", "STR"), keep="first", inplace=True)
    logging.info(f"Removed duplicated (CUI, STR) pairs, {mrconso_df.shape[0]} rows after deletion")

    unique_cuis_set = set(mrconso_df["CUI"].unique())
    logging.info(f"There are {len(unique_cuis_set)} unique CUIs in dataset")
    node_id2cui: Dict[int, str] = {node_id: cui for node_id, cui in enumerate(unique_cuis_set)}
    cui2node_id: Dict[str, int] = {cui: node_id for node_id, cui in node_id2cui.items()}
    assert len(node_id2cui) == len(cui2node_id)
    for _, row in tqdm(mrconso_df.iterrows(), miniters=mrconso_df.shape[0] // 50):
        cui = row["CUI"].strip()
        term_str = row["STR"].strip()
        if term_str == '':
            continue
        node_id = cui2node_id[cui]
        if node_id2terms_list.get(node_id) is None:
            node_id2terms_list[node_id] = []
        node_id2terms_list[node_id].append(term_str.strip())
    logging.info("CUI to terms mapping is created")
    return node_id2terms_list, node_id2cui, cui2node_id


def extract_umls_edges(mrrel_df: pd.DataFrame, cui2node_id: Dict[str, int], ignore_not_mapped_edges=False) \
        -> List[Tuple[int, int]]:
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
    if ignore_not_mapped_edges:
        logging.info(f"{not_mapped_edges_counter} edges are not mapped to any node")
    logging.info(f"Finished generating edges. There are {len(edges)} edges")

    return edges


def extract_umls_oriented_edges_with_relations(mrrel_df: pd.DataFrame, cui2node_id: Dict[str, int],
                                               rel2rel_id: Dict[str, int], rela2rela_id: Dict[str, int],
                                               ignore_not_mapped_edges=False) -> List[Tuple[int, int, int, int]]:
    cuis_relation_str_set = set()
    logging.info("Started generating graph edges")
    edges: List[Tuple[int, int, int, int]] = []
    not_mapped_edges_counter = 0
    for idx, row in tqdm(mrrel_df.iterrows(), miniters=mrrel_df.shape[0] // 100, total=mrrel_df.shape[0]):
        cui_1 = row["CUI1"].strip()
        cui_2 = row["CUI2"].strip()
        rel = row["REL"]
        rela = row["RELA"]
        # Separator validation
        for att in (cui_1, cui_2, rel, rela):
            assert "~~" not in str(att)
        if cui2node_id.get(cui_1) is not None and cui2node_id.get(cui_2) is not None:
            cuis_relation_str = f"{cui_1}~~{cui_2}~~{rel}~~{rela}"
            if cuis_relation_str not in cuis_relation_str_set:
                cui_1_node_id = cui2node_id[cui_1]
                cui_2_node_id = cui2node_id[cui_2]
                rel_id = rel2rel_id[rel]
                rela_id = rela2rela_id[rela]
                edges.append((cui_1_node_id, cui_2_node_id, rel_id, rela_id))
            cuis_relation_str_set.add(cuis_relation_str)
        else:
            if not ignore_not_mapped_edges:
                raise AssertionError(f"Either CUI {cui_1} or {cui_2} are not found in CUI2node_is mapping")
            else:
                not_mapped_edges_counter += 1
    if ignore_not_mapped_edges:
        logging.info(f"{not_mapped_edges_counter} edges are not mapped to any node")
    logging.info(f"Finished generating edges. There are {len(edges)} edges")

    return edges


def add_loops_to_edges_list(node_id2terms_list: Dict[int, List[str]], rel2rel_id: Dict[str, int],
                            rela2rela_id: Dict[str, int], edges: List[Tuple[int, int, int, int]]):
    """
    Takes node_id to terms list mapping and then for each node with more than 1 term(synonyms)
    adds a selp-loop edge to list of edges with "LOOP" relation
    """
    for node_id, terms_list in node_id2terms_list.items():
        if len(terms_list) > 1:
            loop = (node_id, node_id, rel2rel_id["LOOP"], rela2rela_id["LOOP"])
            edges.append(loop)
