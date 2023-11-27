import codecs
import logging
import os
from argparse import ArgumentParser

import pandas as pd


def read_dict(path):
    d = {}
    with codecs.open(path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('||')
            d[attrs[0]] = attrs[1]
    # print(d)
    return d


def read_list(input_path):
    lst = []
    with codecs.open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            lst.append(line.strip())
    return lst


def f(x, d):
    # print(x, d)
    return d[x]


def read_prediction_results(prediction_dir, model_prefix):
    """
    entities.txt
    predictions.txt
    top_1.txt
    true_labels.txt
    """
    entities_path = os.path.join(prediction_dir, "entities.txt")
    predictions_path = os.path.join(prediction_dir, "predictions.txt")
    top_1_path = os.path.join(prediction_dir, "top_1.txt")
    true_labels_path = os.path.join(prediction_dir, "true_labels.txt")

    entities_list = read_list(entities_path)
    predictions_list = read_list(predictions_path)
    top_1_list = read_list(top_1_path)
    true_labels_list = read_list(true_labels_path)

    data_dict = {
        f"{model_prefix}_entities": entities_list,
        f"{model_prefix}_predictions": predictions_list,
        f"{model_prefix}_top_1": top_1_list,
        f"{model_prefix}_true_labels": true_labels_list
    }
    df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
    df[f"{model_prefix}_predictions"] = df[f"{model_prefix}_predictions"].astype("int32")

    return df


def main(args):
    sapbert_predictions_dir = args.sapbert_predictions_dir
    coder_predictions_dir = args.coder_predictions_dir
    gebert_predictions_dir = args.gebert_predictions_dir
    vocabs_dir = args.vocabs_dir
    output_dir = args.output_dir

    for language_code in os.listdir(gebert_predictions_dir):
        gebert_lang_dir = os.path.join(gebert_predictions_dir, language_code)
        for dataset_name in os.listdir(gebert_lang_dir):
            gebert_lang_dataset_dir = os.path.join(gebert_lang_dir, dataset_name)
            sapbert_lang_dataset_dir = os.path.join(sapbert_predictions_dir, language_code, dataset_name)
            coder_lang_dataset_dir = os.path.join(coder_predictions_dir, language_code, dataset_name)
            vocab_path = os.path.join(vocabs_dir, f"mantra_{language_code}_dict_DISO.txt")
            vocab = read_dict(vocab_path)

            print(language_code, dataset_name)

            gebert_df = read_prediction_results(gebert_lang_dataset_dir, "gebert")
            sapbert_df = read_prediction_results(sapbert_lang_dataset_dir, "sapbert")
            coder_df = read_prediction_results(coder_lang_dataset_dir, "coder")

            print("GEBERT\n", gebert_df)
            print("SAPBERT\n", sapbert_df)
            print("CODER\n", coder_df)

            concated_df = pd.concat((gebert_df, sapbert_df, coder_df), axis=1)
            concated_df["gebert_pred_verbose"] = concated_df["gebert_top_1"].apply(lambda x: f(x, vocab))
            concated_df["sapbert_pred_verbose"] = concated_df["sapbert_top_1"].apply(lambda x: f(x, vocab))
            concated_df["coder_pred_verbose"] = concated_df["coder_top_1"].apply(lambda x: f(x, vocab))
            print("CONCATED\n", concated_df)
            concated_df["predictions_gebert_eq_sapbert"] = \
                concated_df["gebert_predictions"] == concated_df["sapbert_predictions"]
            concated_df["predictions_gebert_eq_coder"] = \
                concated_df["gebert_predictions"] == concated_df["coder_predictions"]

            concated_df["predictions_all_equal"] = \
                (concated_df["predictions_gebert_eq_sapbert"] & concated_df["predictions_gebert_eq_coder"])

            sapbert_gebert_not_eq_df = concated_df[~concated_df["predictions_gebert_eq_sapbert"]]
            coder_gebert_not_eq_df = concated_df[~concated_df["predictions_gebert_eq_coder"]]
            three_models_not_eq_df = concated_df[~concated_df["predictions_all_equal"]]
            print("AAA", concated_df["sapbert_predictions"])
            gebert_errors_df = concated_df[concated_df["gebert_predictions"] == 0]

            print("concated\n", concated_df)
            output_subdir = os.path.join(output_dir, language_code, dataset_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            output_path_sapbert_gebert_not_eq = os.path.join(output_subdir, "sapbert_gebert_not_eq.txt")
            output_path_coder_gebert_not_eq = os.path.join(output_subdir, "coder_gebert_not_eq.txt")
            output_path_three_models_not_eq = os.path.join(output_subdir, "three_models_not_eq.txt")
            output_path_gebert_errors = os.path.join(output_subdir, "gebert_errors.txt")
            """
            gebert_entities|
            gebert_predictions
            |gebert_top_1|
            gebert_true_labels
            |sapbert_entities
            |sapbert_predictions
            |sapbert_top_1|
            sapbert_true_labels|
            coder_entities
            |coder_predictions
            |coder_top_1|
            coder_true_labels
            |predictions_gebert_eq_sapbert|
            predictions_gebert_eq_coder|
            predictions_all_equal
            """
            sapbert_gebert_not_eq_df[
                ["gebert_entities", "gebert_predictions", "sapbert_predictions",
                 "gebert_pred_verbose", "sapbert_pred_verbose", ]] \
                .to_csv(output_path_sapbert_gebert_not_eq, sep='|', index=False)
            coder_gebert_not_eq_df[
                ["gebert_entities", "gebert_predictions", "coder_predictions",
                 "gebert_pred_verbose", "coder_pred_verbose"]].to_csv(
                output_path_coder_gebert_not_eq, sep='|', index=False)
            three_models_not_eq_df[
                ["gebert_entities", "gebert_predictions", "sapbert_predictions", "coder_predictions",
                 "gebert_pred_verbose", "sapbert_pred_verbose", "coder_pred_verbose"]].to_csv(
                output_path_three_models_not_eq, sep='|', index=False)
            gebert_errors_df[["gebert_entities", "gebert_predictions", "gebert_pred_verbose", "gebert_top_1", "gebert_true_labels"]].to_csv(
                output_path_gebert_errors, sep='|', index=False)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--sapbert_predictions_dir',
                        default="/home/c204/University/NLP/fair_eval_with_predictions/results/mSAPBERT",
                        type=str)
    parser.add_argument('--coder_predictions_dir',
                        default="/home/c204/University/NLP/fair_eval_with_predictions/results/mCODER",
                        type=str)
    parser.add_argument('--vocabs_dir',
                        default="/home/c204/University/NLP/fair_eval_with_predictions/vocabs",
                        type=str)
    parser.add_argument('--gebert_predictions_dir',
                        default="/home/c204/University/NLP/fair_eval_with_predictions/results/mGEBERT",
                        type=str)
    parser.add_argument('--output_dir', default="error_analysis/", type=str)
    arguments = parser.parse_args()

    main(arguments)
