#!/bin/bash
#SBATCH --job-name=mu_rgcn_dg          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/debug_src.err
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/debug_src.txt
#SBATCH --time=02:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу


nvidia-smi
# --remove_selfloops \
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_rgcn_dgi_sapbert.py --train_dir="/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/GER_GER_FULL/" \
--text_encoder="/home/etutubalina/graph_entity_linking/huggingface_models/xlm-roberta-base/" \
--dataloader_num_workers=0 \
--rgcn_num_hidden_channels 768 \
--rgcn_num_blocks 64 \
--use_rel_or_rela "rel" \
--rgcn_use_fast_conv \
--rgcn_num_neighbors 3 \
--rgcn_num_outer_layers 1 \
--rgcn_num_inner_layers 3 \
--rgcn_dropout_p 0.3 \
--dgi_loss_weight 0.1 \
--remove_selfloops \
--graph_loss_weight 1.0 \
--intermodal_loss_weight 1.0 \
--modality_distance "sapbert" \
--corruption_type "src" \
--text_loss_weight 1.0 \
--use_intermodal_miner \
--intermodal_miner_margin 0.2 \
--max_length=32 \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=3 \
--num_epochs=1 \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="src_debug/"



