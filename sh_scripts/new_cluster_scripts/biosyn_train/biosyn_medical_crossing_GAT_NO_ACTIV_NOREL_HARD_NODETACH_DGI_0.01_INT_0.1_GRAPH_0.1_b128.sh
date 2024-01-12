#!/bin/bash
#SBATCH --job-name=tr_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/train_biosyn_GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/train_biosyn_GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128.txt       # Файл для вывода результатов
#SBATCH --time=48:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1



MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH_NEW/GAT_DGI/gatv2_3_768_1_2_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.01_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_hard_det_txt_False_0.1_lr_2e-05_b_128_NO_ACTIV/checkpoint_e_1_steps_94765.pth/"
NCBID_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/train_dictionary.txt"
NCBID_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_train"

BC5CDRD_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/train_dictionary.txt"
BC5CDRD_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_traindev"

BC5CDRC_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/train_dictionary.txt"
BC5CDRC_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_traindev"

BC2GN_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/train_dictionary.txt"
BC2GN_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_train"

SMM4H_TRAIN_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/train_dictionary.txt"
SMM4H_TRAIN_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/processed_train"

echo "Training NCBI..."
echo "Training NCBI..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/NCBI/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $NCBID_TRAIN_DICT \
--train_dir $NCBID_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/NCBI/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training BC5CDRD..."
echo "Training BC5CDRD..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/BC5CDRD/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $BC5CDRD_TRAIN_DICT \
--train_dir $BC5CDRD_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/BC5CDRD/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training BC5CDRC..."
echo "Training BC5CDRC..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/BC5CDRC/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $BC5CDRC_TRAIN_DICT \
--train_dir $BC5CDRC_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/BC5CDRC/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25



echo "Training BC2GN..."
echo "Training BC2GN..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/BC2GN/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $BC2GN_TRAIN_DICT \
--train_dir $BC2GN_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/BC2GN/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25



echo "Training SMM4H..."
echo "Training SMM4H..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/SMM4H/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $SMM4H_TRAIN_DICT \
--train_dir $SMM4H_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128/SMM4H/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25




