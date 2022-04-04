#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=../logs/create_umls_graph_files.err        # Файл для вывода ошибок
#SBATCH --output=../logs/create_umls_graph_files.txt       # Файл для вывода результатов
#SBATCH --time=09:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python ../scripts/preprocessing/reformat_umls_to_graph.py --mrconso "../UMLS/2020AB/ENG_MRCONSO_filt.RRF" \
--mrrel "../UMLS/2020AB/MRREL.RRF" \
--split_val \
--train_proportion 0.9 \
--output_node_id2synonyms_path "../data/umls_graph/2020AB/ENG_split_not_filtered/synonyms" \
--output_node_id2cui_path "../data/umls_graph/2020AB/ENG_split_not_filtered/id2cui" \
--output_edges_path "../data/umls_graph/2020AB/ENG_split_not_filtered/edges"

