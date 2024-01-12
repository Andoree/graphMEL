#!/bin/bash
#SBATCH --job-name=tr_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/biosyn_medical_crossing_cometa_zeroshot_graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/biosyn_medical_crossing_cometa_zeroshot_graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS.txt       # Файл для вывода результатов
#SBATCH --time=20:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1


COMETA_STRATIFIED_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/cometa_zeroshot/COMETA_id_sf_dictionary.txt"
COMETA_STRATIFIED_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/cometa_zeroshot/train/"





echo "Training graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS..."
echo "Training graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/cometa_zeroshot/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GRAPHSAGE_DGI_MULTILINGUAL/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_0.1_dgi_0.0_modal_sapbert_0.1_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False__lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/" \
--train_dictionary_path $COMETA_STRATIFIED_TRAIN_DICT \
--train_dir $COMETA_STRATIFIED_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/cometa_zeroshot/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25






