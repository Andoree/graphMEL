#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GAT_mBERGAMOR_QUAERO_MEDLINE.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GAT_mBERGAMOR_QUAERO_MEDLINE.txt       # Файл для вывода результатов
#SBATCH --time=23:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES/GAT_DGI_MULTILINGUAL_NO_LOOPS_MAP_ENG_20_20_20/gatv2_3_768_1_3_0.3_2_0.1_graph_l_1.0_rel_NEW_rl_True_dgi_0.1_tl_1.0_inter_sapbert_intermodal_m_True_0.2_rel_feat_False_freeze_False_tl_neigh_False_ilt_sapbert_istrat_None_det_txt_False_1.0_lr_2e-05_b_256_ACTIV__/checkpoint_e_1_steps_119414.pth/"

echo "EUAERO MEDLINE"

python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/MEDLINE/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"



