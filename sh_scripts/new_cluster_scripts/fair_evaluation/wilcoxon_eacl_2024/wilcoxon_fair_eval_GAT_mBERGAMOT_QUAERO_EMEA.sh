#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_mBERGAMOT_QUAERO_EMEA.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_mBERGAMOT_QUAERO_EMEA.txt       # Файл для вывода результатов
#SBATCH --time=23:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/BERGAMOT/GAT-mBERGAMOT/"

echo "QUAERO EMEA FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/EMEA" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"

echo "QUAERO EMEA-fair_exact_vocab"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/EMEA-fair_exact_vocab/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"


echo "QUAERO EMEA-fair_levenshtein_0.2"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/EMEA-fair_levenshtein_0.2/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt"





