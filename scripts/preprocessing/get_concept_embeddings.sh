#!/bin/bash
#SBATCH --job-name=embed_concepts          # Название задачи
#SBATCH --error=logs/get_concept_embeddings.err        # Файл для вывода ошибок
#SBATCH --output=logs/get_concept_embeddings.txt       # Файл для вывода результатов
#SBATCH --time=12:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1
#SBATCH --constraint=type_c|type_b|type_a

python get_concept_embeddings.py --mrconso ../../UMLS/filtered_MRCONSO_SPA.RRF \
--encoder_name "../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--batch_size 512 \
--output_embeddings_path result/spa_embeddings.txt \
--output_vocab_path result/spa_vocab.txt
