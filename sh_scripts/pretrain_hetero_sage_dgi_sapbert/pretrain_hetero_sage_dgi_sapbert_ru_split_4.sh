#!/bin/bash
#SBATCH --job-name=ru_rgcn          # Название задачи
#SBATCH --error=../../logs/pretrain_heterosage_dgi_sapbert_ru_split/ru_pretrain_heterosage_dgi_sapbert_4.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_heterosage_dgi_sapbert_ru_split/ru_pretrain_heterosage_dgi_sapbert_4.txt       # Файл для вывода результатов
#SBATCH --time=10:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python ../../scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--graphsage_num_neighbors=2 \
--num_graphsage_layers=3 \
--graphsage_hidden_channels=768 \
--graphsage_dropout_p=0.2 \
--dgi_loss_weight=0.01 \
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
--output_dir="../../pretrained_graphsapbert/2020AB/Hetero_SAGE/RU_split"

python ../../scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--graphsage_num_neighbors=2 \
--num_graphsage_layers=3 \
--graphsage_hidden_channels=768 \
--graphsage_dropout_p=0.2 \
--dgi_loss_weight=0.01 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=100 \
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
--output_dir="../../pretrained_graphsapbert/2020AB/Hetero_SAGE/RU_split"

python ../../scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--graphsage_num_neighbors=2 \
--num_graphsage_layers=3 \
--graphsage_hidden_channels=768 \
--graphsage_dropout_p=0.2 \
--dgi_loss_weight=0.01 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=64 \
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
--output_dir="../../pretrained_graphsapbert/2020AB/Hetero_SAGE/RU_split"

python ../../scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=4 \
--graphsage_num_neighbors=2 \
--num_graphsage_layers=3 \
--graphsage_hidden_channels=768 \
--graphsage_dropout_p=0.2 \
--dgi_loss_weight=0.01 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=32 \
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
--output_dir="../../pretrained_graphsapbert/2020AB/Hetero_SAGE/RU_split"
