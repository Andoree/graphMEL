#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GraphSAGE_mBERGAMOT_QUAERO_BOTH.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GraphSAGE_mBERGAMOT_QUAERO_BOTH.txt       # Файл для вывода результатов
#SBATCH --time=23:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/BERGAMOT/GraphSAGE-mBERGAMOT/"


echo "QUAERO BOTH FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/BOTH/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"


echo "QUAERO BOTH-fair_exact_vocab"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/BOTH-fair_exact_vocab/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"

echo "QUAERO BOTH-fair_levenshtein_0.2"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/BOTH-fair_levenshtein_0.2/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"



