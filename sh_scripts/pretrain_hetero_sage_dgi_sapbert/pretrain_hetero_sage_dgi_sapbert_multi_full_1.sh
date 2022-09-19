#!/bin/bash
#SBATCH --job-name=mu_hesage          # Название задачи
#SBATCH --error=../../logs/pretrain_heterosage_dgi_sapbert_ru_split/multi_pretrain_heterosage_dgi_sapbert_full.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_heterosage_dgi_sapbert_ru_split/multi_pretrain_heterosage_dgi_sapbert_full.txt       # Файл для вывода результатов
#SBATCH --time=128:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python ../../scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \
--text_encoder="../../models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=4 \
--graphsage_num_neighbors=3 \
--num_graphsage_layers=1 \
--graphsage_hidden_channels=768 \
--graphsage_dropout_p=0.2 \
--filter_rel_types \
--dgi_loss_weight=0.1 \
--remove_selfloops \
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
--output_dir="../../pretrained_graphsapbert/2020AB/Hetero_SAGE/MULTI_full_2"


