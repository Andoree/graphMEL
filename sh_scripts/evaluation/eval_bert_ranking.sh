#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=../../logs/eval_bert_ranking/eval_bert_ranking.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/eval_bert_ranking/eval_bert_ranking.txt       # Файл для вывода результатов
#SBATCH --time=00:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir ../pretrained_encoders/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-5.0-2.5-5_b128_lr2e-05/ \
--data_folder /home/etutubalina/SBERCODE/data/datasets/russian_clinical_data_final/test \
--vocab /home/etutubalina/SBERCODE/data/vocabs/umls_rus_biosyn_fmt.txt
