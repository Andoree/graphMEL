#!/bin/bash
#SBATCH --job-name=map_sem_groups          # Название задачи
#SBATCH --error=../../logs/map_node_ids2sem_group/map_node_ids2sem_group.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/map_node_ids2sem_group/map_node_ids2sem_group.txt       # Файл для вывода результатов
#SBATCH --time=11:20:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу

# English not filtered full
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_pos_pairs_english_notfiltered_FULL" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" &

# English not filtered split
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_pos_pairs_english_notfiltered_SPLIT" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" &

# English filtered full
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_pos_pairs_english_filtered_FULL" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" &

# English filtered split
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_pos_pairs_english_filtered_SPLIT" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" &

# Russian full
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_FULL" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" &

# Russian split
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF"

# Multilingual full
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF"

# Multilingual split
python ../../scripts/preprocessing/map_node_ids2sem_group.py \
--graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_SPLIT" \
--sem_groups_file "/home/etutubalina/classification_transfer_learning/graphmel/UMLS/2020AB/SemGroups_2018.txt" \
--mrsty "../../UMLS/2020AB/MRSTY.RRF"
















