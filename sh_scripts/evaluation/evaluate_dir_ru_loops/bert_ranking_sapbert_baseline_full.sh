#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=../../../logs/evaluate_all_checkpoints_in_dir/node2vec_ru_noloops/eval_sapbert_baseline_full_v2.err        # Файл для вывода ошибок
#SBATCH --output=../../../evaluation/2020AB/Node2vec/RU_split_no_loops/sapbert-baseline-test-full_v2.txt       # Файл для вывода результатов
#SBATCH --time=03:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python ../../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "../../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR_v2/" \
--data_folder "/home/etutubalina/SBERCODE/data/datasets/russian_clinical_data_final/test/" \
--vocab /home/etutubalina/SBERCODE/data/vocabs/umls_rus_biosyn_fmt.txt

