#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=../../../logs/evaluate_all_checkpoints_in_dir/node2vec_ru_noloops/eval_in_dir-in_kb.err        # Файл для вывода ошибок
#SBATCH --output=../../../logs/evaluate_all_checkpoints_in_dir/node2vec_ru_noloops/eval_in_dir-in_kb.txt       # Файл для вывода результатов
#SBATCH --time=23:45:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python ../../../scripts/evaluation/evaluate_all_checkpoints_in_dir.py --data_folder="/home/etutubalina/classification_transfer_learning/graphmel/RuCCoN/final/test-in_kb/" \
--vocab="/home/etutubalina/classification_transfer_learning/graphmel/RuCCoN/umls_rus_biosyn_fmt.txt" \
--input_model_setups_dir="../../../pretrained_encoders/2020AB/Node2vec/RU_split_w_loops" \
--output_evaluation_file_path="../../../evaluation/2020AB/Node2vec/RU_split_no_loops/evaluation_results_test-in_kb.tsv"
