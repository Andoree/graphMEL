#!/bin/bash
#SBATCH --job-name=en_rgcn_dg          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_rgcn_dgi_multilingual_full/dgi_graph_loss_english_pubmedbert_pretrain_rgcn_article_1_layer_sapbert_0.1_dgi_0.0.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_rgcn_dgi_multilingual_full/dgi_graph_loss_english_pubmedbert_pretrain_rgcn_article_1_layer_sapbert_0.1_dgi_0.0.txt       # Файл для вывода результатов
#SBATCH --time=56:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1


nvidia-smi
# --remove_selfloops \
python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_rgcn_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_ENGLISH_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/" \
--dataloader_num_workers=0 \
--rgcn_num_hidden_channels 256 \
--rgcn_num_blocks 64 \
--use_rel_or_rela "rel" \
--rgcn_use_fast_conv \
--rgcn_num_neighbors 3 \
--rgcn_num_outer_layers 1 \
--rgcn_num_inner_layers 3 \
--rgcn_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--remove_selfloops \
--graph_loss_weight 0.1 \
--intermodal_loss_weight 0.1 \
--modality_distance "sapbert" \
--use_intermodal_miner \
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
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/DGI_GRAPH_LOSS_RGCN/ENGLISH_FULL_PUBMEDBERT"



