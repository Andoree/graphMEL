#!/bin/bash
#SBATCH --job-name=ru_hake          # Название задачи
#SBATCH --error=../../logs/pretrain_hake_sapbert/ru_split_pretrain_hake_sapbert_3.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/pretrain_hake_sapbert/ru_split_pretrain_hake_sapbert_3.txt       # Файл для вывода результатов
#SBATCH --time=08:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=6                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi

export TOKENIZERS_PARALLELISM=false
python ../../scripts/self_alignment_pretraining/train_hake_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=6 \
--negative_sample_size=2 \
--hake_gamma=10. \
--hake_modulus_weight=1.0 \
--hake_phase_weight=0.1 \
--hake_adversarial_temperature=1.0 \
--hake_loss_weight=1. \
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
--output_dir="../../pretrained_graphsapbert/2020AB/HAKE_SAPBERT/RU_split"

python ../../scripts/self_alignment_pretraining/train_hake_sapbert.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_SPLIT" \
--validate \
--text_encoder="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=6 \
--negative_sample_size=2 \
--hake_gamma=10. \
--hake_modulus_weight=1.0 \
--hake_phase_weight=0.1 \
--hake_adversarial_temperature=1.0 \
--hake_loss_weight=1. \
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
--output_dir="../../pretrained_graphsapbert/2020AB/HAKE_SAPBERT/RU_split"


