#!/bin/bash
#SBATCH --job-name=ennf_sage          # Название задачи
#SBATCH --error=../../logs/pretrain_graphsage_sapbert_en_full/en_notfiltered_pretrain_graphsage_sapbert_1.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_graphsage_sapbert_en_full/en_notfiltered_pretrain_graphsage_sapbert_1.txt       # Файл для вывода результатов
#SBATCH --time=23:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --nodes=1
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi
python ../../scripts/self_alignment_pretraining/train_graphsage_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_pos_pairs_english_notfiltered_FULL" \
--text_encoder="../../models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/" \
--dataloader_num_workers=4 \
--num_graphsage_channels=768 \
--num_graphsage_layers=1 \
--graphsage_dropout_p=0.5 \
--graphsage_num_neighbors 4 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=256 \
--num_epochs=2 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="../../pretrained_graphsapbert/2020AB/GraphSAGE/EN_NOT_FILTERED_FULL"


