#!/bin/bash
#SBATCH --job-name=links_creation          # Название задачи
#SBATCH --error=logs/create_link_file_for_graphbert.err        # Файл для вывода ошибок
#SBATCH --output=logs/create_link_file_for_graphbert.txt       # Файл для вывода результатов
#SBATCH --time=02:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python create_link_file_for_graphbert.py --mrrel ../../UMLS/MRREL.RRF \
--node2id_path result/all_vocab.txt \
--output_link_path result/link.tsv
