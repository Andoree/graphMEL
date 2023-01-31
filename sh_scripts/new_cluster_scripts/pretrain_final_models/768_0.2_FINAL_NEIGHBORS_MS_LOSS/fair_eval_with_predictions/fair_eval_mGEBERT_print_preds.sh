#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_mGEBERT_print_preds.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/fair_eval_mGEBERT_print_preds.txt       # Файл для вывода результатов
#SBATCH --time=23:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES/GAT_DGI_0.0_MULTILINGUAL_XLMR/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_remove_loops_False_dgi_0.01_text_loss_1.0_intermodal_sapbert_intermodal_miner_True_0.2_relational_features_freeze_neigh_False_True_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_172907.pth/"



echo "dataset mantra/en/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_en_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/en/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt" \
    --output_dir "results/mGEBERT/en/diso_filtered/"


echo "dataset mantra/en/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/en/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt" \
    --output_dir "results/mGEBERT/en/diso_full/"


echo "dataset mantra/en/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/en/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_en_dict_DISO.txt" \
    --output_dir "results/mGEBERT/en/diso_0.2/"


echo "dataset mantra/es/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_es_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" \
    --output_dir "results/mGEBERT/es/diso_filtered/"


echo "dataset mantra/es/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" \
    --output_dir "results/mGEBERT/es/diso_full/"


echo "dataset mantra/es/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/es/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt" \
    --output_dir "results/mGEBERT/es/diso_0.2/"


echo "dataset mantra/nl/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_nl_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" \
    --output_dir "results/mGEBERT/nl/diso_filtered/"


echo "dataset mantra/nl/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" \
    --output_dir "results/mGEBERT/nl/diso_full/"


echo "dataset mantra/nl/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/nl/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_nl_dict_DISO.txt" \
    --output_dir "results/mGEBERT/nl/diso_0.2/"


echo "dataset mantra/fr/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_fr_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
    --output_dir "results/mGEBERT/fr/diso_filtered/"


echo "dataset mantra/fr/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
    --output_dir "results/mGEBERT/fr/diso_full/"


echo "dataset mantra/fr/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/fr/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt" \
    --output_dir "results/mGEBERT/fr/diso_0.2/"

echo "dataset mantra/de/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_de_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" \
    --output_dir "results/mGEBERT/de/diso_filtered/"


echo "dataset mantra/de/ SPLIT DISO"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" \
    --output_dir "results/mGEBERT/de/diso_full/"


echo "dataset mantra/de/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking_print_preds.py --model_dir $MODEL \
    --sapbert_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
    --coder_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all/" \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/mantra/de/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt" \
    --output_dir "results/mGEBERT/de/diso_0.2/"

