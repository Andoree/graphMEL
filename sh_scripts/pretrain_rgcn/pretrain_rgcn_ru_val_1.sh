#!/bin/bash
#SBATCH --job-name=pretrain_graphsage          # Название задачи
#SBATCH --error=../../logs/pretrain_rgcn/ru_val/ru_pretrain_rgcn_1.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_rgcn/ru_val/ru_pretrain_rgcn_1.txt       # Файл для вывода результатов
#SBATCH --time=75:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

python ../../scripts/training/pretrain_rgcn.py --train_dir="../../data/umls_graph/2020AB/RU_split_w_loops/train/" \
--val_dir="../../data/umls_graph/2020AB/RU_split_w_loops/val/" \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--text_encoder_seq_length=32 \
--dataloader_num_workers=2 \
--save_every_N_epoch=1 \
--rgcn_num_hidden_channels 768 \
--rgcn_num_layers=1 \
--rgcn_num_blocks=96 \
--node_neighborhood_sizes 3 \
--use_rel_or_rela="rela" \
--distmult_l2_reg_lambda=0.01 \
--batch_size=256 \
--learning_rate=2e-5 \
--num_epochs=3 \
--random_state=42 \
--gpus=4 \
--output_dir="../../pretrained_models/2020AB/RGCN/RU_split_w_loops"
