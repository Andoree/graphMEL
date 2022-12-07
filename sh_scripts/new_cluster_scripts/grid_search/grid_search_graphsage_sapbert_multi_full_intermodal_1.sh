#!/bin/bash
#SBATCH --job-name=gs_sage          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/grid_search/grid_search_multi_full_graphsage_sapbert_intermodal_1.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/grid_search/grid_search_multi_full_graphsage_sapbert_intermodal_1.txt       # Файл для вывода результатов
#SBATCH --time=48:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=2                   # Требуемое количество GPU
#SBATCH --nodes=1               # Количество используемых узлов
#SBATCH --constraint=type_c|type_b|type_a

#--train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \
export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi

python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/grid_search/graphsage_sapbert_grid_search.py --train_dir="/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/ENG_FRE_GER_SPA_DUT_RUS_MULTILINGUAL_FULL/" \
--text_encoder="/home/etutubalina/graph_entity_linking/huggingface_models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=0 \
--max_length=32 \
--num_inner_graphsage_layers 3 2 \
--graphsage_dropout_p 0.3 \
--graphsage_num_neighbors 2 \
--num_graphsage_channels 256 \
--num_graphsage_layers 2 \
--remove_selfloops \
--graph_loss_weight 1.0 0.1 0.01 \
--text_loss_weight 1.0 \
--intermodal_loss_weight 1.0 0.1 0.01 \
--modality_distance "sapbert" "cosine" \
--batch_size 128 \
--train_subset_ratio 0.01 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--num_epochs=1 \
--data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO" "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO" "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO" "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO" \
--vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
--eval_dataset_name "MANTRA_DE" "MANTRA_ES" "MANTRA_NL" "MANTRA_FR" \
--amp \
--parallel \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="/home/etutubalina/graph_entity_linking/results/grid_search/2020AB/GraphSAGE/gs_MULTI_FULL"

