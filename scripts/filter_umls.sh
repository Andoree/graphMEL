#!/bin/bash
#SBATCH --job-name=train_att          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=00:05:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python filter_umls.py --mrconso "../UMLS/MRCONSO.RRF" \
--mrsty "../UMLS/MRSTY.RRF" \
--save_to "../UMLS/filtered_MRCONSO.RRF" \
--save_all

