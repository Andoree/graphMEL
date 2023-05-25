#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/wilcoxon_fair_eval_graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS.txt       # Файл для вывода результатов
#SBATCH --time=23:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0


echo "NCBI"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/NCBI/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/NCBI/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/NCBI/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/test_dictionary.txt"

echo "bc5cdr-disease"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/BC5CDRD/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC5CDRD" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/BC5CDRD" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/test_dictionary.txt"

echo "bc5cdr-chemical"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/BC5CDRC/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC5CDRC" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/BC5CDRC" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/test_dictionary.txt"

echo "TAC2017ADR"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/TAC2017ADR/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/TAC2017ADR" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/TAC2017ADR" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"


echo "BC2GN"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_wilcoxon.py --model_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/graphsage_DGI_0.0_INTERMODAL_MINER_0.2_REMOVED_LOOPS_NO_N_MS/BC2GN/" \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/BC2GN" \
    --coder_dir "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/BC2GN" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_test_refined" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/test_dictionary.txt"



