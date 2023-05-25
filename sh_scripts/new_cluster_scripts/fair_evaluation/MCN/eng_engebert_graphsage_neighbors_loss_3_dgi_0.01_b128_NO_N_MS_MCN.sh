#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/eng_engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS_MCN.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/eng_engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS_MCN.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GRAPHSAGE_DGI_MULTILINGUAL/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_0.1_dgi_0.01_modal_sapbert_0.1_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False_lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/"

echo "dataset MCN fair_exact_vocab"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test-fair_exact_vocab/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"


echo "dataset MCN"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"


echo "dataset MCN fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"

