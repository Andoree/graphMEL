#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH_NEW/GAT_DGI/gatv2_3_768_1_2_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.01_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_hard_det_txt_False_0.1_lr_2e-05_b_128_NO_ACTIV/checkpoint_e_1_steps_94765.pth/"

echo "NCBI"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/test_dictionary.txt"


echo "bc5cdr-disease"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/test_dictionary.txt"


echo "bc5cdr-chemical"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/test_dictionary.txt"


echo "bc2gm"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/test_dictionary.txt"

echo "smm4h"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/test_dictionary.txt"





