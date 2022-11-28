#!/bin/bash
#SBATCH --job-name=filter_mrconso_cyrillic          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/filter_mrconso_cyrillic/filter_mrconso_cyrillic.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/filter_mrconso_cyrillic/filter_mrconso_cyrillic.txt       # Файл для вывода результатов
#SBATCH --time=05:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --reservation=maint

# Russian full
python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/filter_mrconso_cyrillic.py \
--mrconso "/home/echernyak/graph_entity_linking/UMLS/2020AB/MRCONSO.RRF" \
--output_path "/home/echernyak/graph_entity_linking/UMLS/2020AB/FILTERED_CYRILLIC/MRCONSO.RRF"


















