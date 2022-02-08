#!/bin/bash
#SBATCH --job-name=embed_concepts          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=02:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python create_link_file_for_graphbert.py --mrrel ../../UMLS/MRREL.RRF \
--node2id_path result/vocab.txt \
--output_link_path result/link.tsv
