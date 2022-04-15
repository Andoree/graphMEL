#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=../../../logs/evaluate_all_checkpoints_in_dir/node2vec_ru_loops/eval_ru_loops_in_dir-stratified.err        # Файл для вывода ошибок
#SBATCH --output=../../../logs/evaluate_all_checkpoints_in_dir/node2vec_ru_loops/eval_ru_loops_in_dir-stratified.txt       # Файл для вывода результатов
#SBATCH --time=23:45:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python ../../../scripts/evaluation/evaluate_all_checkpoints_in_dir.py --data_folder="/home/etutubalina/SBERCODE/data/datasets/russian_clinical_data_final/test-stratified" \
--vocab="/home/etutubalina/SBERCODE/data/vocabs/umls_rus_biosyn_fmt.txt" \
--input_model_setups_dir="../../../pretrained_encoders/2020AB/Node2vec/RU_split_w_loops_for_sure/" \
--output_evaluation_file_path="../../../evaluation/2020AB/Node2vec/RU_split_loops_for_sure/evaluation_results_test-stratified.tsv"
