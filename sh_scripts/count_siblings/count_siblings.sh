#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=../../logs/count_siblings_mrrel/create_umls_graph_files.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/count_siblings_mrrel/create_umls_graph_files.txt       # Файл для вывода результатов
#SBATCH --time=03:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python ../../scripts/statistics/count_siblings.py --mrrel "../../UMLS/2020AB/MRREL.RRF" \
--output_path "../../stats/count_siblings/count_siblings_mrrel.txt"

