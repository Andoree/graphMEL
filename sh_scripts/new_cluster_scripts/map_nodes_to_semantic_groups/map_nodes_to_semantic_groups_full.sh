#!/bin/bash
#SBATCH --job-name=map_sem_groups          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/map_node_ids2sem_group/map_node_ids2sem_group.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/map_node_ids2sem_group/map_node_ids2sem_group.txt       # Файл для вывода результатов
#SBATCH --time=11:20:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

# English not filtered full
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_ENGLISH_FULL/" \
--sem_groups_file "/home/etutubalina/graph_entity_linking/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRSTY.RRF" &

# Russian full
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL" \
--sem_groups_file "/home/etutubalina/graph_entity_linking/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRSTY.RRF"

# Multilingual full
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_FRE_GER_SPA_DUT_RUS_MULTILINGUAL_FULL" \
--sem_groups_file "/home/etutubalina/graph_entity_linking/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRSTY.RRF"

















