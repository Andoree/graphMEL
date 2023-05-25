#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_RGCN_DGI_0.01_768_0.2_FINAL_NO_N_MS_ENGLISH.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_RGCN_DGI_0.01_768_0.2_FINAL_NO_N_MS_ENGLISH.txt       # Файл для вывода результатов
#SBATCH --time=03:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/RGCN_DGI_MULTILINGUAL/dgi_0.01_rgcn_1_3_[3]_text_1.0_remove_loops_Truegraph_loss_0.1_intermodal_sapbert_0.1_0.3_768--None-64_rel_intermodal_miner_True_0.2_freeze_neigh_False_text_loss_neighbors_False_lr_2e-05_b_128_fast_rgcn_conv/checkpoint_e_1_steps_94765.pth/"


echo "dataset mantra/en/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_en_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/en/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"


echo "dataset mantra/en/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/en/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"


echo "dataset mantra/en/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/en/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"


echo "dataset MCN fair_exact_vocab"
echo VOCAB "mantra_en_dict_DISO.txt"
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


