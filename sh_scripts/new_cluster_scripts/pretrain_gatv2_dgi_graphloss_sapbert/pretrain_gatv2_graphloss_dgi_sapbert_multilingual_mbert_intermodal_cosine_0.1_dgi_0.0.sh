#!/bin/bash
#SBATCH --job-name=mu_gat_dg          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_gatv2_dgi_multilingual_full/mbert_dgi_graph_loss_multilingual_pretrain_gatv2_mbert_article_1_layer_cosine_0.1_dgi_0.0.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_gatv2_dgi_multilingual_full/mbert_dgi_graph_loss_multilingual_pretrain_gatv2_mbert_article_1_layer_cosine_0.1_dgi_0.0.txt       # Файл для вывода результатов
#SBATCH --time=56:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1

nvidia-smi
# --remove_selfloops \
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_gatv2_dgi_sapbert.py --train_dir="/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_FRE_GER_SPA_DUT_RUS_MULTILINGUAL_FULL/" \
--text_encoder="/home/etutubalina/graph_entity_linking/huggingface_models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=0 \
--gat_num_outer_layers 1 \
--gat_num_inner_layers 3 \
--gat_num_hidden_channels 256 \
--gat_num_neighbors 3 \
--gat_num_att_heads 2 \
--gat_dropout_p 0.3 \
--gat_attention_dropout_p 0.1 \
--use_rel_or_rela "rel" \
--graph_loss_weight 0.1 \
--dgi_loss_weight 0. \
--remove_selfloops \
--use_intermodal_miner \
--intermodal_loss_weight 0.1 \
--modality_distance "cosine" \
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
--output_dir="/home/etutubalina/graph_entity_linking/results/pretrained_graphsapbert/2020AB/DGI_GRAPH_LOSS_GatV2/MULTILINGUAL_FULL_MBERT"



