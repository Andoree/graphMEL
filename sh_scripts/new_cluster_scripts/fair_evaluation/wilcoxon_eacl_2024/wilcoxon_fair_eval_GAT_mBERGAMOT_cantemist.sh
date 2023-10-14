#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_mBERGAMOT_cantemist.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_mBERGAMOT_cantemist.txt       # Файл для вывода результатов
#SBATCH --time=23:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/BERGAMOT/GAT-mBERGAMOT/"

echo "CANTEMIST BOTH FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist_fixed_format/cantemist-norm-concepts" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"

echo "CANTEMIST BOTH-fair_exact_vocab"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist_fixed_format/cantemist-norm-concepts-fair_exact_vocab/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "CANTEMIST BOTH-fair_levenshtein_0.2"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist_fixed_format/cantemist-norm-concepts-fair_levenshtein_0.2/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"





