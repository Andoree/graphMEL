#!/bin/bash
#SBATCH --job-name=mu_hesage          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/pretrain_dgi_0.0_graph_loss_HETERO_GRAPHSAGE_xlmr_all_languages_full_final_sapbert_intermodal_nominer_768.err         # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/graphmel/logs/pretrain_graph_models_final/pretrain_dgi_0.0_graph_loss_HETERO_GRAPHSAGE_xlmr_all_languages_full_final_sapbert_intermodal_nominer_768.txt       # Файл для вывода результатов
#SBATCH --time=150:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=5                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1


export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python /home/echernyak/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_hetero_graphsage_dgi_sapbert.py --train_dir="/home/echernyak/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_SPA_POR_FRE_JPN_RUS_DUT_GER_ITA_CZE_SWE_KOR_LAV_HUN_CHI_NOR_POL_TUR_EST_FIN_SCR_UKR_GRE_DAN_BAQ_HEB_MULTILINGUAL_ALL_LANGUAGES_FULL/" \
--text_encoder="/home/echernyak/graph_entity_linking/huggingface_models/xlm-roberta-base/" \
--dataloader_num_workers=0 \
--graphsage_num_neighbors 1 \
--num_graphsage_layers 3 \
--graphsage_hidden_channels 768 \
--graphsage_dropout_p 0.3 \
--graph_loss_weight 0.1 \
--dgi_loss_weight 0.0 \
--intermodal_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
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
--output_dir="/home/echernyak/graph_entity_linking/results/pretrained_graphsapbert/2020AB/FINAL_MODELS/HETERO_GRAPHSAGE_DGI_0.0_MULTILINGUAL_ALL_LANGUAGES"


