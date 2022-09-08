#!/bin/bash
#SBATCH --job-name=ru_rgcn          # Название задачи
#SBATCH --error=../../logs/grid_search/grid_search_multi_full_rgcn_dgi_sapbert.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/grid_search/grid_search_multi_full_rgcn_dgi_sapbert.txt       # Файл для вывода результатов
#SBATCH --time=23:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

#--train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \

nvidia-smi
python ../../scripts/grid_search/rgcn_dgi_sapbert_grid_search.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/RUS_pos_pairs_russian_FULL" \
--validate \
--text_encoder="../../models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=4 \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--num_epochs=1 \
--data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/de/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/es/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/nl/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/fr/DISO" \
--vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
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
--output_dir="../../grid_search/2020AB/RGCN_DGI/gs_MULTI_FULL"

