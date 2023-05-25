#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_RGCN_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_CODIESP_D.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_RGCN_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_CODIESP_D.txt       # Файл для вывода результатов
#SBATCH --time=23:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES/RGCN_DGI_0.0_MULTILINGUAL_XLMR/dgi_0.01_rgcn_1_3_[3]_text_1.0_remove_loops_Truegraph_loss_0.1_intermodal_sapbert_0.1_0.3_768--None-64_rel_intermodal_miner_True_0.2_freeze_neigh_False_lr_2e-05_b_128_fast_rgcn_conv/checkpoint_e_1_steps_172907.pth/"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT fair_exact_vocab"
echo VOCAB "codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"




