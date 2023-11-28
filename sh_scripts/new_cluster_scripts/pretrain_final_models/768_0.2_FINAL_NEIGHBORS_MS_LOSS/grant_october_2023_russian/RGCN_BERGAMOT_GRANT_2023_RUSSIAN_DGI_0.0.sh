#!/bin/bash
#SBATCH --job-name=ru_rgcn_dg          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/RGCN_BERGAMOT_GRANT_2023_RUSSIAN_DGI_0.0.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/RGCN_BERGAMOT_GRANT_2023_RUSSIAN_DGI_0.0.txt       # Файл для вывода результатов
#SBATCH --time=15:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1


nvidia-smi
# --remove_selfloops \
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_rgcn_dgi_sapbert.py --train_dir="/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL/" \
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
--dgi_loss_weight 0.0 \
--remove_selfloops \
--graph_loss_weight 1.0 \
--intermodal_loss_weight 1.0 \
--modality_distance "sapbert" \
--corruption_type "trg" \
--text_loss_weight 1.0 \
--use_intermodal_miner \
--intermodal_miner_margin 0.2 \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=1 \
--amp \
--parallel \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="/home/etutubalina/graph_entity_linking/results/pretrained_graphsapbert/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_RUS_GRANT_2023/RGCN_DGI_RUS/"



