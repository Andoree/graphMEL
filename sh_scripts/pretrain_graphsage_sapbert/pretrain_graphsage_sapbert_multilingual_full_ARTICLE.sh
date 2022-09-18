#!/bin/bash
#SBATCH --job-name=mul_sage          # Название задачи
#SBATCH --error=../../logs/pretrain_graphsage_sapbert_multilingual_full/multilingual_pretrain_graphsage_sapbert_article.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_graphsage_sapbert_multilingual_full/multilingual_pretrain_graphsage_sapbert_article.txt       # Файл для вывода результатов
#SBATCH --time=40:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=5                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi
python ../../scripts/self_alignment_pretraining/train_graphsage_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \
--text_encoder="../../models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=4 \
--num_graphsage_channels=768 \
--num_graphsage_layers=2 \
--num_inner_graphsage_layers 1 \
--graphsage_dropout_p=0.1 \
--graphsage_num_neighbors 2 \
--graph_loss_weight=1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
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
--output_dir="../../pretrained_graphsapbert/2020AB/GraphSAGE/MULTILINGUAL_FULL"


