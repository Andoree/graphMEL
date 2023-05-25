#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/eng_coder_baseline_MCN.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/eng_coder_baseline_MCN.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/"

echo "dataset MCN fair_exact_vocab"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test-fair_exact_vocab/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"


echo "dataset MCN"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"


echo "dataset MCN fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/MCN_n2c2/biosyn_processed_pairs/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/SNOMEDCT_US-all-aggregated.txt"

