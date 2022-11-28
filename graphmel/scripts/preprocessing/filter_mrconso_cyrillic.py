import codecs
import logging
import os.path
from argparse import ArgumentParser
import random
import numpy as np
from typing import Dict

import pandas as pd

from graphmel.scripts.utils.io import read_mrconso

RU_LETTERS_SET = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')


def string_has_ru_letters(str_value: str) -> bool:
    if str_value is not np.nan:
        char_set = set(str_value.lower())
        ru_letters_intersection = char_set.intersection(RU_LETTERS_SET)
        if len(ru_letters_intersection) > 0:
            return True
    return False


def main(args):
    random.seed(42)

    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    mrconso_df = read_mrconso(fpath=args.mrconso)

    mrconso_df["has_ru_letters"] = mrconso_df["STR"].apply(string_has_ru_letters)
    mrconso_df["has_ru_letters"] = mrconso_df[mrconso_df["has_ru_letters"]]
    mrconso_df.drop(columns=("has_ru_letters",), inplace=True)

    pd.read_csv(output_path, header=False, sep='|', encoding='utf-8', quoting=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso', type=str, default=f"C:/University/NLP/UMLS/2020AB-full/2020AB/META/MRCONSO.RRF")
    parser.add_argument('--output_path', type=str, default="delete.RRF")

    arguments = parser.parse_args()
    main(arguments)
