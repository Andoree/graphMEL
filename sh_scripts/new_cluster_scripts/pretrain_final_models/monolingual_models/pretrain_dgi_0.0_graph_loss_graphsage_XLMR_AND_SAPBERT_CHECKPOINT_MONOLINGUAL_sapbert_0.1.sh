#!/bin/bash
#SBATCH --job-name=mul_sage          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_monolingual_graphsage_graph_0.1_dgi_0.0/pretrain_dgi_0.0_graph_loss_graphsage_XLMR_AND_SAPBERT_CHECKPOINT_MONOLINGUAL_sapbert_0.1.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_monolingual_graphsage_graph_0.1_dgi_0.0/pretrain_dgi_0.0_graph_loss_graphsage_XLMR_AND_SAPBERT_CHECKPOINT_MONOLINGUAL_sapbert_0.1.txt       # Файл для вывода результатов
#SBATCH --time=38:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/SPA_SPA_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/SPA_FULL_SAPBERT_CHECKPOINT"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/DUT_DUT_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/DUT_FULL_SAPBERT_CHECKPOINT"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/FRE_FRE_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/FRE_FULL_SAPBERT_CHECKPOINT"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/GER_GER_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/GER_FULL_SAPBERT_CHECKPOINT"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL_CYRILLIC/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/RUS_CYRILLIC_FULL_SAPBERT_CHECKPOINT"


python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/SPA_SPA_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/SPA_FULL_XLMR"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/DUT_DUT_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/DUT_FULL_XLMR"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/FRE_FRE_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/FRE_FULL_XLMR"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/GER_GER_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/GER_FULL_XLMR"

python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/RUS_RUSSIAN_FULL_CYRILLIC/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base" \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=3 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=3 \
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/GRAPHSAGE_GRAPH_DGI_0.0/RUS_CYRILLIC_FULL_XLMR"


