#!/bin/bash
#SBATCH --job-name=gs_hake          # Название задачи
#SBATCH --error=../../logs/grid_search/grid_search_multi_full_hake_sapbert_2_step2.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/grid_search/grid_search_multi_full_hake_sapbert_2_step2.txt       # Файл для вывода результатов
#SBATCH --time=72:00:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=4               # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

nvidia-smi
# --remove_selfloops \
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ../../scripts/grid_search/hake_sapbert_grid_search.py --train_dir="../../data/umls_graph/2020AB_pos_pairs_datasets/ENG_FRE_GER_SPA_DUT_RUS_pos_pairs_multilingual_FULL" \
--text_encoder="../../models/bert-base-multilingual-uncased/" \
--dataloader_num_workers=4 \
--data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/de/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/es/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/nl/DISO" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/fr/DISO" \
--vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
--eval_dataset_name "MANTRA_DE" "MANTRA_ES" "MANTRA_NL" "MANTRA_FR" \
--negative_sample_size 4 \
--hake_gamma 10 \
--hake_modulus_weight 1. \
--hake_phase_weight 0. \
--hake_adversarial_temperature 0.3 \
--hake_loss_weight 0.1 \
--filter_transitive_relations no \
--filter_semantic_type_nodes yes no \
--batch_size 128 \
--mrsty "../../UMLS/2020AB/MRSTY.RRF" \
--train_subset_ratio 0.001 \
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
--output_dir="../../grid_search/2020AB/HAKE_Sapbert/gs_MULTI_FULL_2_step2"



