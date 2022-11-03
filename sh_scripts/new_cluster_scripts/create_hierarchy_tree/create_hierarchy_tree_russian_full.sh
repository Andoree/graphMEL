#!/bin/bash
#SBATCH --job-name=rus_hie_tree          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/create_hierarchy_tree_dataset/create_hierarchy_tree_russian_full.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/create_hierarchy_tree_dataset/create_hierarchy_tree_russian_full.txt       # Файл для вывода результатов
#SBATCH --time=04:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/extract_tree_from_graph_dataset.py \
--mrsty "/home/echernyak/graph_entity_linking/UMLS/2020AB/MRSTY.RRF" \
--input_graph_dataset_dir "/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL/" \
--output_dir "/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL_TREE/"

