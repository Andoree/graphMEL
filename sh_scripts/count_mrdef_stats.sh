#!/bin/bash
#SBATCH --job-name=count_concept_defintions          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=00:20:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../scripts/statistics/count_concept_descriptions.py --mrconso "../UMLS/MRCONSO.RRF" \
--mrdef "../UMLS/MRDEF.RRF" \
--output_path "../scripts/statistics/stats.txt"

