#!/bin/bash
#SBATCH --job-name=mul_sage          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graphsage_sapbert_multilingual_full/multilingual_pretrain_graphsage_sapbert_article_1_layer_sapbert_0.1.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graphsage_sapbert_multilingual_full/multilingual_pretrain_graphsage_sapbert_article_1_layer_sapbert_0.1.txt       # Файл для вывода результатов
#SBATCH --time=40:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python /home/echernyak/graph_entity_linking/graphmel/scripts/self_alignment_pretraining/train_graphsage_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_FRE_GER_SPA_DUT_RUS_MULTILINGUAL_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=0 \
--num_graphsage_channels=256 \
--num_graphsage_layers=1 \
--num_inner_graphsage_layers 3 \
--graphsage_dropout_p 0.3 \
--graphsage_num_neighbors 3 \
--text_loss_weight 1.0 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--intermodal_loss_weight 0.1 \
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
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/GraphSAGE/MULTILINGUAL_FULL_MBERT"


