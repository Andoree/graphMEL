#!/bin/bash
#SBATCH --job-name=create_pos_pairs          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/create_pos_pairs_dataset/create_pos_pairs_dataset_full_FRE.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/create_pos_pairs_dataset/create_pos_pairs_dataset_full_FRE.txt       # Файл для вывода результатов
#SBATCH --time=09:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/create_positive_triplets_dataset.py --mrconso "/home/echernyak/graph_entity_linking/UMLS/2020AB/MRCONSO.RRF" \
--mrrel "/home/echernyak/graph_entity_linking/UMLS/2020AB/MRREL.RRF" \
--langs "FRE" \
--output_dir "/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/FRE_FULL/"

