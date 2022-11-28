#!/bin/bash
#SBATCH --job-name=mul_sage          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/xlmr_dgi_0.01_graph_loss_0.1_all_languages_pretrain_graphsage_xlmr_sapbert_intermodal_miner_0.4_768.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/xlmr_dgi_0.01_graph_loss_0.1_all_languages_pretrain_graphsage_xlmr_sapbert_intermodal_miner_0.4_768.txt       # Файл для вывода результатов
#SBATCH --time=37:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_FRE_GER_SPA_DUT_RUS_MULTILINGUAL_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.01 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--use_intermodal_miner \
--intermodal_miner_margin 0.4 \
--text_loss_weight 1.0 \
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
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS_MINER_0.4/GRAPHSAGE_DGI_0.0_MULTILINGUAL"


