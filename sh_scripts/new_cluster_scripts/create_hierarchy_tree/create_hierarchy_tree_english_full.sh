#!/bin/bash
#SBATCH --job-name=rus_hie_tree          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/create_hierarchy_tree_dataset/create_hierarchy_tree_english_full.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/create_hierarchy_tree_dataset/create_hierarchy_tree_english_full.txt       # Файл для вывода результатов
#SBATCH --time=04:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/extract_tree_from_graph_dataset.py \
--mrsty "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRSTY.RRF" \
--input_graph_dataset_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_ENGLISH_FULL_TREE/" \
--output_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_ENGLISH_FULL_TREE/"

