#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_USE_REL_FEATURES_EN_MANTRA.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_USE_REL_FEATURES_EN_MANTRA.txt       # Файл для вывода результатов
#SBATCH --time=09:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES/GAT_DGI_0.0_MULTILINGUAL_XLMR/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_remove_loops_False_dgi_0.01_text_loss_1.0_intermodal_sapbert_intermodal_miner_True_0.2_relational_features_freeze_neigh_False_True_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_172907.pth/"


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


