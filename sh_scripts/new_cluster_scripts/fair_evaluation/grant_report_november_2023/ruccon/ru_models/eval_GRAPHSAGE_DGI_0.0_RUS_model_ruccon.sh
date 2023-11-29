#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_ruccon/baselines/eval_GRAPHSAGE_DGI_0.0_RUS_model_ruccon.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_ruccon/baselines/eval_GRAPHSAGE_DGI_0.0_RUS_model_ruccon.txt       # Файл для вывода результатов
#SBATCH --time=04:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1

MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_RUS_GRANT_2023/GRAPHSAGE_DGI_RUS/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_1.0_dgi_0.0_modal_sapbert_1.0_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False__lr_2e-05_b_128_trg/checkpoint_e_1_steps_2770.pth/"

echo "STRATIFIED TEST"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir $MODEL \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-stratified" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

echo "TEST ZERO SHOT"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir $MODEL \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-zero_shot" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"




