#!/bin/bash
#SBATCH --job-name=pretrain_node2vec          # Название задачи
#SBATCH --error=../logs/pretrain_node2vec.err        # Файл для вывода ошибок
#SBATCH --output=../logs/pretrain_node2vec.txt       # Файл для вывода результатов
#SBATCH --time=23:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi

python ../scripts/training/pretrain_node2vec.py --train_node2terms_path="../data/umls_graph/2020AB/RU/train_synonyms" \
--train_edges_path="../data/umls_graph/2020AB/RU/train_edges" \
--val_node2terms_path="../data/umls_graph/2020AB/RU/val_synonyms" \
--val_edges_path="../data/umls_graph/2020AB/RU/val_edges" \
--text_encoder="../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--text_encoder_seq_length=32 \
--dataloader_num_workers=8 \
--node2vec_num_negative_samples=1 \
--node2vec_context_size=5 \
--node2vec_walks_per_node=1 \
--node2vec_p=3. \
--node2vec_q=2. \
--node2vec_walk_length=7 \
--batch_size=128 \
--learning_rate=2e-5 \
--num_epochs=10 \
--random_state=42 \
--save_every_N_epoch=5 \
--gpus=4 \
--output_dir="../pretrained_models/2020AB/Node2vec/RU_split"

nvidia-smi
