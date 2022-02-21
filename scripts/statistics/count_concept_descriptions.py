import codecs
import os.path
from argparse import ArgumentParser

import pandas as pd

from graphmel.scripts.utils.io import read_mrconso, save_dict
from graphmel.scripts.utils.io import read_mrdef


def calc_concept_definition_stats(mrconso_df: pd.DataFrame, mrdef_df: pd.DataFrame, groupby_sab=False):
    unique_concept_ids = set(mrconso_df["CUI"].unique())
    unique_concept_ids_with_defs = set(mrdef_df["CUI"].unique())
    concept_def_intersection = unique_concept_ids.intersection(unique_concept_ids_with_defs)
    res = {
        "# Unique CUIs": len(unique_concept_ids),
        "# Concepts with definitions": len(unique_concept_ids_with_defs),
        "Concepts with definitions proportion": len(concept_def_intersection) / len(unique_concept_ids)
    }
    if groupby_sab:
        mrconso_df["has_def"] = mrconso_df["CUI"].apply(lambda x: 1 if x in unique_concept_ids_with_defs else 0)
        unique_sabs_list = mrconso_df["SAB"].unique()
        cui_with_def_stats_by_sab = {}
        for source_name in unique_sabs_list:
            source_subset_df = mrconso_df[mrconso_df["SAB"] == source_name]
            unique_source_cuis = set(source_subset_df["CUI"].unique())
            unique_source_cuis_with_def = set(source_subset_df[source_subset_df["has_def"] == 1]["CUI"].unique())

            cui_with_def_stats_by_sab[source_name] = len(unique_source_cuis_with_def) / len(unique_source_cuis)
        for k, v in sorted(cui_with_def_stats_by_sab.items(), key=lambda item: -item[1]):
            res[f"Concepts with definition proportion, {k}"] = v

    return res


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrdef')
    parser.add_argument('--groupby_sab', action='store_true')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    groupby_sab = args.groupby_sab
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df_mrconso = read_mrconso(args.mrconso)
    df_mrdef = read_mrdef(args.mrdef)
    concepts_with_def_stats = calc_concept_definition_stats(mrconso_df=df_mrconso, mrdef_df=df_mrdef,
                                                            groupby_sab=groupby_sab)
    save_dict(save_path=output_path, dictionary=concepts_with_def_stats, )


if __name__ == '__main__':
    main()
