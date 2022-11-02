#!/bin/bash
#SBATCH --job-name=mu_hesage          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_heterosage_graphloss_dgi_multilingual_full/xlmr_dgi_graph_loss_multilingual_pretrain_hetero_sage_xlmr_article_1_layer_cosine_0.1_dgi_0.01.err         # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_heterosage_graphloss_dgi_multilingual_full/xlmr_dgi_graph_loss_multilingual_pretrain_hetero_sage_xlmr_article_1_layer_cosine_0.1_dgi_0.01.txt       # Файл для вывода результатов
#SBATCH --time=128:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python ../../scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_FRE_GER_SPA_DUT_RUS_MULTILINGUAL_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base/" \
--dataloader_num_workers=0 \
--graphsage_num_neighbors 3 \
--num_graphsage_layers 3 \
--graphsage_hidden_channels 256 \
--graphsage_dropout_p 0.3 \
--graph_loss_weight 0.1 \
--dgi_loss_weight 0.01 \
--intermodal_loss_weight 0.1 \
--modality_distance "cosine" \
--text_loss_weight 1.0 \
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
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/DGI_GRAPH_LOSS_HETERO_GRAPHSAGE/MULTILINGUAL_FULL_XLMR"


