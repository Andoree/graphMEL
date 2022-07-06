#!/bin/bash
#SBATCH --job-name=ru_gat          # Название задачи
#SBATCH --error=../../logs/pretrain_gat_dgi_sapbert_ru_split/ru_pretrain_gat_dgi_sapbert_9.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_gat_dgi_sapbert_ru_split/ru_pretrain_gat_dgi_sapbert_9.txt       # Файл для вывода результатов
#SBATCH --time=05:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi
# --remove_selfloops \
python ../../scripts/self_alignment_pretraining/train_gatv2_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--gat_num_hidden_channels=64 \
--gat_num_neighbors 3 \
--gat_num_att_heads=12 \
--gat_attention_dropout_p=0.1 \
--gat_use_relation_features \
--remove_selfloops \
--gat_edge_dim=64 \
--use_rel_or_rela="rel" \
--dgi_loss_weight=1e-3 \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=256 \
--num_epochs=2 \
--amp \
--parallel \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="../../pretrained_graphsapbert/2020AB/GATv2_DGI/RU_split"

python ../../scripts/self_alignment_pretraining/train_gatv2_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--gat_num_hidden_channels=64 \
--gat_num_neighbors 3 \
--gat_num_att_heads=12 \
--gat_attention_dropout_p=0.1 \
--gat_use_relation_features \
--remove_selfloops \
--gat_edge_dim=64 \
--use_rel_or_rela="rel" \
--dgi_loss_weight=1e-3 \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=2 \
--amp \
--parallel \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="../../pretrained_graphsapbert/2020AB/GATv2_DGI/RU_split"


