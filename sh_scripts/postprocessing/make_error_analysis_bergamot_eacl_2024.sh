#!/bin/bash
#SBATCH --job-name=error_analysis          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/postprocessing/make_error_analysis_bergamot_eacl_2024.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/postprocessing/make_error_analysis_bergamot_eacl_2024.txt       # Файл для вывода результатов
#SBATCH --time=05:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу


python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/postprocessing/make_error_analysis.py \
--sapbert_predictions_dir /home/etutubalina/graph_entity_linking/graphmel/sh_scripts/new_cluster_scripts/fair_evaluation/fair_eval_with_predictions_eacl_2024/results/mSAPBERT/ \
--coder_predictions_dir /home/etutubalina/graph_entity_linking/graphmel/sh_scripts/new_cluster_scripts/fair_evaluation/fair_eval_with_predictions_eacl_2024/results/mCODER/ \
--gebert_predictions_dir /home/etutubalina/graph_entity_linking/graphmel/sh_scripts/new_cluster_scripts/fair_evaluation/fair_eval_with_predictions_eacl_2024/results/mGEBERT/ \
--vocabs_dir "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/" \
--output_dir "/home/etutubalina/graph_entity_linking/graphmel/sh_scripts/new_cluster_scripts/fair_evaluation/fair_eval_with_predictions_eacl_2024/error_analysis/"

