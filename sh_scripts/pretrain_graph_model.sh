#!/bin/bash
#SBATCH --job-name=pretrain_graph_model          # Название задачи
#SBATCH --error=../logs/pretrain_graph_model.err        # Файл для вывода ошибок
#SBATCH --output=../logs/pretrain_graph_model.txt       # Файл для вывода результатов
#SBATCH --time=23:59:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU

python ../scripts/training/pretrain_graph_model.py --train_node2terms_path="../../data/umls_graph/2020AB/RU/train_synonyms" \
--train_edges_path="../../data/umls_graph/2020AB/RU/train_edges" \
--val_node2terms_path="../../data/umls_graph/2020AB/RU/val_synonyms" \
--val_edges_path="../../data/umls_graph/2020AB/RU/val_edges" \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--text_encoder_seq_length=32 \
--graphsage_num_layers=2 \
--graphsage_num_channels=256 \
--graph_num_neighbors=5 \
--graphsage_dropout= \
--random_walk_length=2 \
--batch_size=32 \
--learning_rate=2e-5 \
--num_epochs=2 \
--debug \
--random_state=42 \
--gpus 2 \
--output_dir="../../pretrained_models/2020AB/RU"

