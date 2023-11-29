#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_ruccon/baselines/eval_hetero_graphsage_DGI_0.0_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_ruccon.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_ruccon/baselines/eval_hetero_graphsage_DGI_0.0_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_ruccon.txt       # Файл для вывода результатов
#SBATCH --time=04:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1

MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES/DGI_GRAPH_LOSS_HETERO_GRAPHSAGE/MULTILINGUAL_FULL_XLMR_TO_ENG_20_20_20/graphsage_n-[1]_l-3_c-256_p-0.3_text_1.0_graph_1.0_intermodal_sapbert_1.0_dgi_0.1_intermodal_miner_True_freeze_neigh_False_lr_2e-05_b_256/checkpoint_e_1_steps_119414.pth/"

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




