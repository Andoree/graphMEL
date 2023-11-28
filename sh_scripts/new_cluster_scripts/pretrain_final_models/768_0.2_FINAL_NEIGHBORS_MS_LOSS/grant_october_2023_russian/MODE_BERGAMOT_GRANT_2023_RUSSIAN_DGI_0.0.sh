#!/bin/bash
#SBATCH --job-name=ru_hake          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_hake_sapbert/MODE_BERGAMOT_GRANT_2023_RUSSIAN_DGI_0.0.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/pretrain_hake_sapbert/MODE_BERGAMOT_GRANT_2023_RUSSIAN_DGI_0.0.txt       # Файл для вывода результатов
#SBATCH --time=15:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_hake_sapbert.py --train_dir="/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL_TREE" \
--text_encoder="/home/etutubalina/graph_entity_linking/huggingface_models/xlm-roberta-base/" \
--dataloader_num_workers=0 \
--negative_sample_size 4 \
--hake_gamma 10.0 \
--hake_modulus_weight 1.0 \
--hake_phase_weight 0.0 \
--hake_adversarial_temperature 1.0 \
--hake_loss_weight 0.1 \
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
--output_dir="/home/etutubalina/graph_entity_linking/results/pretrained_graphsapbert/2020AB/HAKE_SAPBERT/MODE_RUS_GRANT_2023/"



