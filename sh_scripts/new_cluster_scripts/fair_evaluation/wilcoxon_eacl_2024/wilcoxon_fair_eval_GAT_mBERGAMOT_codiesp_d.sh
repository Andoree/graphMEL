#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_mBERGAMOT_codiesp_d.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_mBERGAMOT_codiesp_d.txt       # Файл для вывода результатов
#SBATCH --time=23:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/BERGAMOT/GAT-mBERGAMOT/"

echo "CODIESP-D BOTH FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"

echo "CODIESP-D BOTH-fair_exact_vocab"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_exact_vocab/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "CODIESP-D BOTH-fair_levenshtein_0.2"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_levenshtein_0.2/" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"





