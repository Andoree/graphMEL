import codecs
import os.path
from argparse import ArgumentParser
from typing import Dict, Tuple

import pandas as pd

from graphmel.scripts.utils.io import read_mrconso, save_dict
from graphmel.scripts.utils.io import read_mrdef


def calc_concept_definition_stats(mrconso_df: pd.DataFrame, mrdef_df: pd.DataFrame, groupby_sab=False) \
        -> Tuple[pd.DataFrame, Dict]:
    unique_concept_ids = set(mrconso_df["CUI"].unique())
    unique_concept_ids_with_defs = set(mrdef_df["CUI"].unique())
    concept_def_intersection = unique_concept_ids.intersection(unique_concept_ids_with_defs)
    global_stats_dict = {
        "# Unique CUIs": len(unique_concept_ids),
        "# Concepts with definitions": len(unique_concept_ids_with_defs),
        "Concepts with definitions proportion": len(concept_def_intersection) / len(unique_concept_ids)
    }
    by_dataset_stats_df = None
    entries = []
    if groupby_sab:
        mrconso_df["has_def"] = mrconso_df["CUI"].apply(lambda x: 1 if x in unique_concept_ids_with_defs else 0)
        unique_sabs_list = mrconso_df["SAB"].unique()
        cui_with_def_stats_by_sab = {}
        unique_cui_stats_by_source = {}
        for source_name in unique_sabs_list:
            source_subset_df = mrconso_df[mrconso_df["SAB"] == source_name]
            unique_source_cuis = set(source_subset_df["CUI"].unique())
            unique_source_cuis_with_def = set(source_subset_df[source_subset_df["has_def"] == 1]["CUI"].unique())

            cui_with_def_stats_by_sab[source_name] = len(unique_source_cuis_with_def) / len(unique_source_cuis)
            unique_cui_stats_by_source[source_name] = len(unique_source_cuis)
        for source_name, cui_with_def_proportion in sorted(cui_with_def_stats_by_sab.items(),
                                                           key=lambda item: -item[1]):
            stats_dict = {
                "source_name": source_name,
                f"% with definition": cui_with_def_proportion,
                f"# Unique CUIs": unique_cui_stats_by_source[source_name]
            }
            entries.append(stats_dict)
        by_dataset_stats_df = pd.DataFrame(entries)

    return by_dataset_stats_df, global_stats_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrdef')
    parser.add_argument('--groupby_sab', action='store_true')
    parser.add_argument('--groupby_stats_output_path')
    parser.add_argument('--global_stats_output_path')
    args = parser.parse_args()

    groupby_sab = args.groupby_sab
    global_stats_output_path = args.global_stats_output_path
    groupby_stats_output_path = args.groupby_stats_output_path
    output_paths_list = (global_stats_output_path, groupby_stats_output_path)
    for output_path in output_paths_list:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)

    df_mrconso = read_mrconso(args.mrconso)
    df_mrdef = read_mrdef(args.mrdef)
    concepts_with_def_stats_df, global_stats_dict = calc_concept_definition_stats(mrconso_df=df_mrconso, mrdef_df=df_mrdef,
                                                                           groupby_sab=groupby_sab)
    if concepts_with_def_stats_df is not None:
        concepts_with_def_stats_df.to_csv(groupby_stats_output_path, sep='\t', index=False)
    save_dict(save_path=global_stats_output_path, dictionary=global_stats_dict, )


if __name__ == '__main__':
    main()
