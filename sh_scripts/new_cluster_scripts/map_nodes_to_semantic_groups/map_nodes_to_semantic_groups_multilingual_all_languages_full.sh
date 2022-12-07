#!/bin/bash
#SBATCH --job-name=map_sem_groups          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/map_node_ids2sem_group/map_node_ids2sem_group_multilingual_all_languages_full.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/map_node_ids2sem_group/map_node_ids2sem_group_multilingual_all_languages_full.txt       # Файл для вывода результатов
#SBATCH --time=05:45:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --reservation=maint

# Multilingual all languages full
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_SPA_POR_FRE_JPN_RUS_DUT_GER_ITA_CZE_SWE_KOR_LAV_HUN_CHI_NOR_POL_TUR_EST_FIN_SCR_UKR_GRE_DAN_BAQ_HEB_MULTILINGUAL_ALL_LANGUAGES_FULL_TREE" \
--sem_groups_file "/home/etutubalina/graph_entity_linking/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRSTY.RRF"

















