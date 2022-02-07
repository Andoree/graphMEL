#!/bin/bash
#SBATCH --job-name=embed_concepts          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=03:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python get_concept_embeddings.py --mrconso ../../UMLS/MRCONSO_test.RRF \
--encoder_name "../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--output_embeddings_path embeddings.txt \
--output_vocab_path vocab.txt
