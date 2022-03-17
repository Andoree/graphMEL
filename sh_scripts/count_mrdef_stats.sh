#!/bin/bash
#SBATCH --job-name=count_concept_defintions          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=00:20:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../scripts/statistics/count_concept_descriptions.py --mrconso "../UMLS/2020AB/MRCONSO.RRF" \
--groupby_sab \
--mrdef "../UMLS/2020AB/MRDEF.RRF" \
--groupby_stats_output_path "../statistics/stats_2020AB.txt" \
--global_stats_output_path "../statistics/stats_2020AB.txt"
