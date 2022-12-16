#!/bin/bash
#SBATCH --job-name=mul_sage          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/pretrain_dgi_0.01_graph_loss_graphsage_xlmr_full_final_sapbert_intermodal_768_0.2_FINAL_ALL_LANGUAGES_b_192.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/pretrain_dgi_0.01_graph_loss_graphsage_xlmr_full_final_sapbert_intermodal_768_0.2_FINAL_ALL_LANGUAGES_b_192.txt       # Файл для вывода результатов
#SBATCH --time=37:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python3.8 /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_SPA_POR_FRE_JPN_RUS_DUT_GER_ITA_CZE_SWE_KOR_LAV_HUN_CHI_NOR_POL_TUR_EST_FIN_SCR_UKR_GRE_DAN_BAQ_HEB_MULTILINGUAL_ALL_LANGUAGES_FULL/" \
--text_encoder="/home/etutubalina/graph_entity_linking/huggingface_models/xlm-roberta-base/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.01 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--text_loss_weight 1.0 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--use_intermodal_miner \
--intermodal_miner_margin 0.2 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=192 \
--num_epochs=1 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="/home/etutubalina/graph_entity_linking/results/pretrained_graphsapbert/2020AB/768_0.2_FINAL_ALL_LANGUAGES_b_192/GRAPHSAGE_DGI_ALL_LANGUAGES"


