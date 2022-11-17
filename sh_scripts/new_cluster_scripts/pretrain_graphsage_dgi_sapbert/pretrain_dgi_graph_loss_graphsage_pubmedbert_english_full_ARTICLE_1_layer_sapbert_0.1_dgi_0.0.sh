#!/bin/bash
#SBATCH --job-name=mdSAXL         # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graphsage_dgi_multilingual_full/pubmedbert_english_checkpoint_dgi_graph_loss_pretrain_graphsage_sapbert_checkpoint_article_1_layer_sapbert_0.1_dgi_0.0.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graphsage_dgi_multilingual_full/pubmedbert_english_checkpoint_dgi_graph_loss_pretrain_graphsage_sapbert_checkpoint_article_1_layer_sapbert_0.1_dgi_0.0.txt       # Файл для вывода результатов
#SBATCH --time=48:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_ENGLISH_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 256 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.00 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--use_intermodal_miner \
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
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/DGI_GRAPH_LOSS_GraphSAGE/ENGLISH_FULL_PUBMEDBERT"


