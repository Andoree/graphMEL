#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_refined_cometa_zeroshot.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_refined_cometa_zeroshot.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
DICT_PATH="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/cometa_zeroshot/COMETA_id_sf_dictionary.txt"
DATA_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/cometa_zeroshot/processed_test_refined/"


MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/"
echo "zeroshot cometa sapbert refined"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder $DATA_DIR \
    --vocab $DICT_PATH


MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/"
echo "zeroshot cometa coder refined"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder $DATA_DIR \
    --vocab $DICT_PATH


MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GRAPHSAGE_DGI_MULTILINGUAL/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_0.1_dgi_0.0_modal_sapbert_0.1_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False__lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/"
echo "zeroshot cometa graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS refined"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder $DATA_DIR \
    --vocab $DICT_PATH


MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GAT_DGI_MULTILINGUAL/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.0_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_None_det_txt_False_0.1_lr_2e-05_b_128_ACTIV__/checkpoint_e_1_steps_94765.pth/"
echo "zeroshot cometa GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP refined"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder $DATA_DIR \
    --vocab $DICT_PATH


