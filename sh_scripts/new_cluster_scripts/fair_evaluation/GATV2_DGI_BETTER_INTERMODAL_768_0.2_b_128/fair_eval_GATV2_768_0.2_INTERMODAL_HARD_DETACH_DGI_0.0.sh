#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GATV2_768_0.2_INTERMODAL_HARD_DETACH_DGI_0.0.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_GATV2_768_0.2_INTERMODAL_HARD_DETACH_DGI_0.0.txt       # Файл для вывода результатов
#SBATCH --time=03:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_WEIGHTED_INTERMODAL_ALL_LANGUAGES/GAT_DGI_WEIGHTED_INTERMODAL_MULTILINGUAL/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_1.0_rel_NEW_rl_False_dgi_0.0_tl_1.0_inter_sapbert_intermodal_miner_False_0.2_relational_features_True_freeze_neigh_False_tl_neighbors_False_ilt_cosine_istrat_hard_det_txt_True_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_172907.pth/"


echo "dataset mantra/es/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_es_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"


echo "dataset mantra/es/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"


echo "dataset mantra/es/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"


echo "dataset mantra/nl/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_nl_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"


echo "dataset mantra/nl/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"


echo "dataset mantra/nl/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"


echo "dataset mantra/fr/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_fr_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"


echo "dataset mantra/fr/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"


echo "dataset mantra/fr/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"


echo "dataset mantra/de/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_de_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"


echo "dataset mantra/de/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"


echo "dataset mantra/de/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"

