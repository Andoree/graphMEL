#!/bin/bash
#SBATCH --job-name=node2vec_dataloader_debug          # Название задачи
#SBATCH --error=../../logs/debug/node2vec_dataloader_debug.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/debug/node2vec_dataloader_debug.txt       # Файл для вывода результатов
#SBATCH --time=23:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --constraint=type_c|type_b|type_a

python ../../scripts/training/node2vec_dataset_test.py --node2terms_path="../../data/umls_graph/2020AB/RU/val_synonyms" \
--edges_path="../../data/umls_graph/2020AB/RU/val_edges" \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=2