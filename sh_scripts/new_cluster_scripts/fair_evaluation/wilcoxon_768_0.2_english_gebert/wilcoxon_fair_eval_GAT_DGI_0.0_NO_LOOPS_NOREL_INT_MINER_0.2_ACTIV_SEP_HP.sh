#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP.txt       # Файл для вывода результатов
#SBATCH --time=23:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0


echo "NCBI"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/test_dictionary.txt"

echo "bc5cdr-disease"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/test_dictionary.txt"

echo "bc5cdr-chemical"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/test_dictionary.txt"

echo "TAC2017ADR"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/TAC2017ADR/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"


echo "BC2GN"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/test_dictionary.txt"



