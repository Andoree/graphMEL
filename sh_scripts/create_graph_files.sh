#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=../logs/create_umls_graph_files.err        # Файл для вывода ошибок
#SBATCH --output=../logs/create_umls_graph_files.txt       # Файл для вывода результатов
#SBATCH --time=00:40:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../scripts/preprocessing/reformat_umls_to_graph.py --mrconso "../UMLS/2020AA/filt_MRCONSO_SPA_ENG_RUS_DUT_FRE_GER.RRF" \
--mrrel "../UMLS/2020AA/MRREL.RRF" \
--split_val \
--train_proportion 0.9 \
--output_node_id2synonyms_path "../data/umls_graph/2020AA/synonyms" \
--output_node_id2cui_path "../data/umls_graph/2020AA/id2cui" \
--output_edges_path "../data/umls_graph/2020AA/edges"

