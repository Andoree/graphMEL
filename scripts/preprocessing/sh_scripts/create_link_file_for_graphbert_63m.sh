#!/bin/bash
#SBATCH --job-name=links_creation          # Название задачи
#SBATCH --error=../logs/create_link_file_for_graphbert_63m.err        # Файл для вывода ошибок
#SBATCH --output=../logs/create_link_file_for_graphbert_63m.txt       # Файл для вывода результатов
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../create_link_file_for_graphbert.py --mrrel ../../../UMLS/MRREL_63m.RRF \
--node2id_path ../result/all_vocab.txt \
--output_link_path ../result/link_63m.tsv
