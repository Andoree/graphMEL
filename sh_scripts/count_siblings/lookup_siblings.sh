#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=../../logs/count_siblings_mrrel/siblings_lookup.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/count_siblings_mrrel/siblings_lookup.txt       # Файл для вывода результатов
#SBATCH --time=03:50:00                      # Максимальное время выполнения
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python ../../scripts/statistics/siblings_lookup.py --mrrel "../../UMLS/2020AB/MRREL.RRF" \
--output_path "../../stats/count_siblings/count_siblings_mrrel.txt"

