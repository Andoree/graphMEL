#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/sapbert_fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ENGLISH_USE_REL_FEATURES_MCN.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/sapbert_fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ENGLISH_USE_REL_FEATURES_MCN.txt       # Файл для вывода результатов
#SBATCH --time=10:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=6                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ENGLISH/GAT_DGI_ENGLISH/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_remove_loops_False_dgi_0.01_text_loss_1.0_intermodal_sapbert_intermodal_miner_True_0.2_relational_features_freeze_neigh_False_text_loss_neighbors_True_True_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/"



echo "dataset MCN fair_exact_vocab"
python eval.py --model_dir $MODEL \
--dictionary_path "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt" \
--data_dir "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test-fair_exact_vocab/" \
--use_cuda \
--output_dir "fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ENGLISH_USE_REL_FEATURES_MCN/exact" \
--custom_query_loader \
--max_length 25 \
--agg_mode cls

echo "dataset MCN FULL"
python eval.py --model_name_or_path $MODEL \
--dictionary_path "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt" \
--data_dir "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test/" \
--use_cuda \
--output_dir "fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ENGLISH_USE_REL_FEATURES_MCN/full" \
--custom_query_loader \
--max_length 25 \
--agg_mode cls


echo "dataset MCN fair_levenshtein_0.2"
python eval.py --model_name_or_path $MODEL \
--dictionary_path "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt" \
--data_dir "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test-fair_levenshtein_0.2" \
--use_cuda \
--output_dir "fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ENGLISH_USE_REL_FEATURES_MCN/leven_0.2" \
--custom_query_loader \
--max_length 25 \
--agg_mode cls







