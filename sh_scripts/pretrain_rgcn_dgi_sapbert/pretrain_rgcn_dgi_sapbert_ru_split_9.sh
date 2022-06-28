#!/bin/bash
#SBATCH --job-name=ru_rgcn          # Название задачи
#SBATCH --error=../../logs/pretrain_rgcn_dgi_sapbert_ru_split/ru_pretrain_rgcn_dgi_sapbert_9.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_rgcn_dgi_sapbert_ru_split/ru_pretrain_rgcn_dgi_sapbert_9.txt       # Файл для вывода результатов
#SBATCH --time=10:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi
python ../../scripts/self_alignment_pretraining/train_rgcn_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--rgcn_num_hidden_channels=768 \
--rgcn_num_neighbors 3 \
--rgcn_num_blocks=96 \
--use_rel_or_rela="rel" \
--dgi_loss_weight=1 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=512 \
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
--output_dir="../../pretrained_graphsapbert/2020AB/RGCN_DGI/RU_split"


python ../../scripts/self_alignment_pretraining/train_rgcn_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--rgcn_num_hidden_channels=768 \
--rgcn_num_neighbors 3 \
--rgcn_num_blocks=96 \
--use_rel_or_rela="rel" \
--dgi_loss_weight=1 \
--remove_selfloops \
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
--output_dir="../../pretrained_graphsapbert/2020AB/RGCN_DGI/RU_split"

python ../../scripts/self_alignment_pretraining/train_rgcn_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--rgcn_num_hidden_channels=768 \
--rgcn_num_neighbors 3 \
--rgcn_num_blocks=96 \
--use_rel_or_rela="rel" \
--dgi_loss_weight=1 \
--remove_selfloops \
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
--output_dir="../../pretrained_graphsapbert/2020AB/RGCN_DGI/RU_split"