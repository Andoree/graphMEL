#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_encoder.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_encoder.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/"

echo "NCBI"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/test_dictionary.txt"


echo "bc5cdr-disease"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/test_dictionary.txt"


echo "bc5cdr-chemical"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/test_dictionary.txt"


echo "bc2gm"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/test_dictionary.txt"

echo "smm4h"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/test_dictionary.txt"





