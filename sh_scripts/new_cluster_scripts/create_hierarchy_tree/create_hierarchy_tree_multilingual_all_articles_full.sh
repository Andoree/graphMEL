#!/bin/bash
#SBATCH --job-name=rus_hie_tree          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/create_hierarchy_tree_dataset/create_hierarchy_tree_multilingual_all_articles_full.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/create_hierarchy_tree_dataset/create_hierarchy_tree_multilingual_all_atricles_full.txt       # Файл для вывода результатов
#SBATCH --time=04:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --reservation=maint


python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/extract_tree_from_graph_dataset.py \
--mrsty "/home/echernyak/graph_entity_linking/UMLS/2020AB/MRSTY.RRF" \
--input_graph_dataset_dir "/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_SPA_POR_FRE_JPN_RUS_DUT_GER_ITA_CZE_SWE_KOR_LAV_HUN_CHI_NOR_POL_TUR_EST_FIN_SCR_UKR_GRE_DAN_BAQ_HEB_MULTILINGUAL_ALL_LANGUAGES_FULL_TREE/" \
--output_dir "/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_SPA_POR_FRE_JPN_RUS_DUT_GER_ITA_CZE_SWE_KOR_LAV_HUN_CHI_NOR_POL_TUR_EST_FIN_SCR_UKR_GRE_DAN_BAQ_HEB_MULTILINGUAL_ALL_LANGUAGES_FULL_TREE/"

