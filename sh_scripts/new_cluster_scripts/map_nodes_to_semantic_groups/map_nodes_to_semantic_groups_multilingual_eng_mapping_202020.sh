#!/bin/bash
#SBATCH --job-name=map_sem_groups          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/map_node_ids2sem_group/map_nodes_to_semantic_groups_multilingual_eng_mapping_202020.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/map_node_ids2sem_group/map_nodes_to_semantic_groups_multilingual_eng_mapping_202020.txt       # Файл для вывода результатов
#SBATCH --time=15:20:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

# English not filtered full
python /home/etutubalina/graph_entity_linking/graphmel/scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_SPA_POR_FRE_JPN_RUS_DUT_GER_ITA_CZE_SWE_KOR_LAV_HUN_CHI_NOR_POL_TUR_EST_FIN_SCR_UKR_GRE_DAN_BAQ_HEB_MULTILINGUAL_ALL_LANGUAGES_MAP_TO_ENG_20_20_20_FULL/" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRSTY.RRF"


















