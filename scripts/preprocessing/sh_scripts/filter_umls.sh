#!/bin/bash
#SBATCH --job-name=filter_umls          # Название задачи
#SBATCH --error=logs/filter_umls.err        # Файл для вывода ошибок
#SBATCH --output=logs/filter_umls.txt       # Файл для вывода результатов
#SBATCH --time=00:12:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python filter_umls.py --mrconso "../../UMLS/MRCONSO.RRF" \
--mrsty "../../UMLS/MRSTY.RRF" \
--langs "SPA" \
--filter_unique_str \
--save_to "../../UMLS/filtered_MRCONSO_SPA.RRF" \
--save_all

