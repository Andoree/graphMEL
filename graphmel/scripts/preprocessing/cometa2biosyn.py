import codecs
import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    input_data_dir = args.input_data_dir
    output_dir = args.output_dir


    for dataset in ("train", "dev", "test"):
        input_dataset_path = os.path.join(input_data_dir, f"{dataset}.csv")
        output_data_dir = os.path.join(output_dir, f"{dataset}/")
        output_file_path = os.path.join(output_data_dir, "0.concept")
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        data_df = pd.read_csv(input_dataset_path, sep='\t')
        with codecs.open(output_file_path, "w+", encoding="utf-8") as out_file:
            for i, row in data_df.iterrows():
                term = row["Term"]
                concept_id = row["General SNOMED ID"]
                out_file.write(f"-1||0|0||Disease||{term}||{concept_id}\n")







if __name__ == '__main__':
    main()
