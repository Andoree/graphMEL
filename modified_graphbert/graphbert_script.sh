#!/bin/bash
#SBATCH --job-name=embed_concepts          # Название задачи
#SBATCH --error=logs/graph_bert_embeddings_training.err        # Файл для вывода ошибок
#SBATCH --output=logs/graph_bert_embeddings_training.txt       # Файл для вывода результатов
#SBATCH --time=12:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=1
#SBATCH --constraint=type_c|type_b|type_a

python graph_bert_embeddings_training.py --graph_size 9513524 \
--feature_size 768 \
--use_cuda \
--dataset_name umls \
--preprocess \
--k_list 1 2 3 4 5 \
--max_hop_k 5
