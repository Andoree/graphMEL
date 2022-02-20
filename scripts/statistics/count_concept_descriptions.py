import codecs
import os.path
from argparse import ArgumentParser

import pandas as pd

from graphMEL.scripts.utils.io import read_mrconso
from graphMEL.scripts.utils.io import read_mrdef


def calc_concept_definition_stats(mrconso_df: pd.DataFrame, mrdef_df: pd.DataFrame):
    num_unique_concepts = set(mrconso_df["CUI"].unique())
    num_concepts_with_defs = set(mrdef_df["CUI"].unique())
    res = {
        "# Unique CUIs": len(num_unique_concepts),
        "# Concepts with definitions": len(num_concepts_with_defs),
        "# Concepts with definitions proportion": len(num_concepts_with_defs) / len(num_unique_concepts)
    }
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrdef')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df_mrconso = read_mrconso(args.mrconso)
    df_mrdef = read_mrdef(args.mrdef)
    print(df_mrdef)
    d = calc_concept_definition_stats(mrconso_df=df_mrconso, mrdef_df=df_mrdef)
    # TODO: Refactoring
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for key, val in d.items():
            out_file.write(f"{key}:\t{val}\n")


if __name__ == '__main__':
    main()
