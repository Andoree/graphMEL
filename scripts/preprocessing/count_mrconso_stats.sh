#!/bin/bash
#SBATCH --job-name=job          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=00:05:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python count_mrconso_stats.py --mrconso "../../UMLS/MRCONSO.RRF" \
--output_path "mrconso_stats.txt" 

