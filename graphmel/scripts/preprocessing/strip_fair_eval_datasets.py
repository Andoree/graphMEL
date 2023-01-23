import codecs
import logging
import os
from argparse import ArgumentParser


def main(args):
    input_base_dir = args.input_base_dir
    input_dataset_dirs = args.input_dataset_dirs
    output_base_dir = args.output_base_dir

    for inp_dataset_dir in input_dataset_dirs:
        full_input_dataset_dir = os.path.join(input_base_dir, inp_dataset_dir)
        full_output_dataset_dir = os.path.join(output_base_dir, inp_dataset_dir)
        for filename in os.listdir(full_input_dataset_dir):
            input_data_filename_path = os.path.join(full_input_dataset_dir, filename)
            output_data_filename_path = os.path.join(full_output_dataset_dir, filename)
            if not os.path.exists(output_data_filename_path) and output_data_filename_path != '':
                os.makedirs(output_data_filename_path)
            with codecs.open(input_data_filename_path, 'r', encoding="utf-8") as inp_file, \
                    codecs.open(output_data_filename_path, 'r', encoding="utf-8") as out_file:
                for line in inp_file:
                    line = line.strip().rstrip('|')
                    out_file.write(f"{line}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_base_dir', type=str,
                        default="/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/")
    parser.add_argument('--input_dataset_dirs', type=str, nargs='+',
                        default=["test-fair_exact_vocab/", "test", "test-fair_levenshtein_0.2"])
    parser.add_argument('--output_base_dir', type=str,
                        default="/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/new_biosyn_processed_pairs/")

    arguments = parser.parse_args()
    main(arguments)

    main(arguments)