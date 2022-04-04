#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=../logs/create_umls_graph_files_5.err        # Файл для вывода ошибок
#SBATCH --output=../logs/create_umls_graph_files_5.txt       # Файл для вывода результатов
#SBATCH --time=04:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../scripts/preprocessing/reformat_umls_to_graph.py --mrconso "../UMLS/2020AB/ENG_MRCONSO_filt.RRF" \
--mrrel "../UMLS/2020AB/MRREL.RRF" \
--output_node_id2synonyms_path "../data/umls_graph/2020AB/FULL_EN/synonyms" \
--output_node_id2cui_path "../data/umls_graph/2020AB/FULL_EN/id2cui" \
--output_edges_path "../data/umls_graph/2020AB/FULL_EN/edges"

