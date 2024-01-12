#!/bin/bash
#SBATCH --job-name=tr_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/biosyn_medical_crossing_cometa_stratified.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/biosyn_medical_crossing_cometa_stratified.txt       # Файл для вывода результатов
#SBATCH --time=20:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1


COMETA_STRATIFIED_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/cometa_stratified/COMETA_id_sf_dictionary.txt"
COMETA_STRATIFIED_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/cometa_stratified/train/"



echo "Training EnSapBERT..."
echo "Training EnSapBERT..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/cometa_stratified/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
--train_dictionary_path $COMETA_STRATIFIED_TRAIN_DICT \
--train_dir $COMETA_STRATIFIED_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/cometa_stratified/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training EnCODER..."
echo "Training EnCODER..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/cometa_stratified/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/" \
--train_dictionary_path $COMETA_STRATIFIED_TRAIN_DICT \
--train_dir $COMETA_STRATIFIED_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/cometa_stratified/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS..."
echo "Training graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/cometa_stratified/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GRAPHSAGE_DGI_MULTILINGUAL/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_0.1_dgi_0.0_modal_sapbert_0.1_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False__lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/" \
--train_dictionary_path $COMETA_STRATIFIED_TRAIN_DICT \
--train_dir $COMETA_STRATIFIED_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/cometa_stratified/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP..."
echo "Training GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/cometa_stratified/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GAT_DGI_MULTILINGUAL/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.0_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_None_det_txt_False_0.1_lr_2e-05_b_128_ACTIV__/checkpoint_e_1_steps_94765.pth/" \
--train_dictionary_path $COMETA_STRATIFIED_TRAIN_DICT \
--train_dir $COMETA_STRATIFIED_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/cometa_stratified/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25







