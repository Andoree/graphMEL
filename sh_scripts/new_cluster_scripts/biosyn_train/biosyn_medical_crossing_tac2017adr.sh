#!/bin/bash
#SBATCH --job-name=tr_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/biosyn_medical_crossing_tac2017adr.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/biosyn_medical_crossing_tac2017adr.txt       # Файл для вывода результатов
#SBATCH --time=20:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1


TAC_ADR_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/train_dictionary.txt"
TAC_ADR_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_train"



echo "Training EnSapBERT..."
echo "Training EnSapBERT..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/TAC_ADR/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
--train_dictionary_path $TAC_ADR_TRAIN_DICT \
--train_dir $TAC_ADR_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/TAC_ADR/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training EnCODER..."
echo "Training EnCODER..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/TAC_ADR/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/" \
--train_dictionary_path $TAC_ADR_TRAIN_DICT \
--train_dir $TAC_ADR_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/TAC_ADR/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25

echo "Training engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS..."
echo "Training engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS/TAC_ADR/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GRAPHSAGE_DGI_MULTILINGUAL/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_0.1_dgi_0.01_modal_sapbert_0.1_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False_lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/" \
--train_dictionary_path $TAC_ADR_TRAIN_DICT \
--train_dir $TAC_ADR_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS/TAC_ADR/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training GAT_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1..."
echo "Training GAT_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1/TAC_ADR/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH_NEW/GAT_DGI/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.01_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_hard_det_txt_False_0.1_lr_2e-05_b_128_ACTIV/checkpoint_e_1_steps_94765.pth/" \
--train_dictionary_path $TAC_ADR_TRAIN_DICT \
--train_dir $TAC_ADR_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1/TAC_ADR/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


