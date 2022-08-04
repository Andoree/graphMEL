#!/bin/bash
#SBATCH --job-name=rus_hie_tree          # Название задачи
#SBATCH --error=../../logs/create_hierarchy_tree/create_pos_pairs_dataset_multilingual_split.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/create_hierarchy_tree/create_pos_pairs_dataset_multilingual_split.txt       # Файл для вывода результатов
#SBATCH --time=04:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python ../../scripts/preprocessing/extract_tree_from_graph_dataset.py \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" \
--input_graph_dataset_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_SPLIT" \
--output_dir "../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_SPLIT"

