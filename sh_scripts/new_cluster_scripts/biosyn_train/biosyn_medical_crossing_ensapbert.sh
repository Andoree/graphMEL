#!/bin/bash
#SBATCH --job-name=tr_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/train_biosyn_ensapbert.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_train/train_biosyn_ensapbert.txt       # Файл для вывода результатов
#SBATCH --time=48:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1



MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/"
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
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/NCBI/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $NCBID_TRAIN_DICT \
--train_dir $NCBID_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/NCBI/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training BC5CDRD..."
echo "Training BC5CDRD..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC5CDRD/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $BC5CDRD_TRAIN_DICT \
--train_dir $BC5CDRD_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC5CDRD/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25


echo "Training BC5CDRC..."
echo "Training BC5CDRC..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC5CDRC/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $BC5CDRC_TRAIN_DICT \
--train_dir $BC5CDRC_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC5CDRC/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25



echo "Training BC2GN..."
echo "Training BC2GN..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC2GN/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $BC2GN_TRAIN_DICT \
--train_dir $BC2GN_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC2GN/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25



echo "Training SMM4H..."
echo "Training SMM4H..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/SMM4H/"
python /home/etutubalina/graph_entity_linking/BioSyn/train.py \
--model_name_or_path $MODEL \
--train_dictionary_path $SMM4H_TRAIN_DICT \
--train_dir $SMM4H_TRAIN_DIR \
--output_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/SMM4H/" \
--use_cuda \
--topk 20 \
--epoch 20 \
--train_batch_size 16 \
--learning_rate 1e-5 \
--max_length 25




