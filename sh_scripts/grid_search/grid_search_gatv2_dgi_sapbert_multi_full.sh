#!/bin/bash
#SBATCH --job-name=gs_gat          # Название задачи
#SBATCH --error=../../logs/grid_search/grid_search_multi_full_gatv2_dgi_sapbert.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/grid_search/grid_search_multi_full_gatv2_dgi_sapbert.txt       # Файл для вывода результатов
#SBATCH --time=23:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4               # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi
# --remove_selfloops \
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ../../scripts/grid_search/gatv2_dgi_sapbert_grid_search.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \
--text_encoder="../../models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=4 \
--data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/de/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/es/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/nl/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/fr/DISO" \
--vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
--eval_dataset_name "MANTRA_DE" "MANTRA_ES" "MANTRA_NL" "MANTRA_FR" \
--gat_num_neighbors 2 3 4 \
--gat_num_hidden_channels 768 \
--gat_num_layers 1 3 5 \
--gat_dropout_p 0.1 0.3 \
--gat_num_att_heads 1 2 \
--gat_attention_dropout_p 0.1 0.3 \
--gat_use_relation_features true false \
--gat_edge_dim 16 32 \
--dgi_loss_weight 1. 0.1 0.01 \
--batch_size 128 96 \
--train_subset_ratio 0.0001 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
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
--output_dir="../../grid_search/2020AB/GATv2_DGI/gs_MULTI_FULL"



