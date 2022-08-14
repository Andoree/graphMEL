#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=../../logs/fair_evaluation/fair_eval_mbert_rgcn_dgi.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/fair_evaluation/fair_eval_mbert_rgcn_dgi.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1

export CUDA_VISIBLE_DEVICES=0
MODEL = "/home/etutubalina/classification_transfer_learning/graphmel/pretrained_graph_sapbert_encoders/2020AB/RGCN_DGI_SapBERT/MULTILINGUAL_FULL/dgi_0.001_rgcn_[5]_768--None-96_rellr_2e-05_b_128_rgcn_conv/checkpoint_e_1_steps_159512.pth"


echo "dataset mantra/es/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_es_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/es/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"


echo "dataset mantra/es/ SPLIT DISO"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/es/DISO" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"


echo "dataset mantra/es/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/es/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"


echo "dataset mantra/de/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_de_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/de/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"


echo "dataset mantra/de/ SPLIT DISO"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/de/DISO" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"


echo "dataset mantra/de/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/de/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_de_dict_DISO.txt"


echo "dataset mantra/fr/ SPLIT DISO-fair_exact_vocab"
echo VOCAB "mantra_fr_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/fr/DISO-fair_exact_vocab" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"


echo "dataset mantra/fr/ SPLIT DISO"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/fr/DISO" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"


echo "dataset mantra/fr/ SPLIT DISO-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/mantra/fr/DISO-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/mantra_fr_dict_DISO.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test"
echo VOCAB "codiesp-d-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_exact"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_exact" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_exact_vocab"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_exact_vocab" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_levenshtein_train_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_levenshtein_train_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test"
echo VOCAB "codiesp-p-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_exact"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_exact" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_exact_vocab"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_exact_vocab" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_levenshtein_train_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_levenshtein_train_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts"
echo VOCAB "CANTEMIST-lopez-ubeda-et-al.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_exact"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_exact" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_exact_vocab"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_exact_vocab" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_levenshtein_train_0.2"
echo VOCAB "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python ../../Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_levenshtein_train_0.2" \
    --vocab "/home/etutubalina/classification_transfer_learning/graphmel/data/medical_crossing_data/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


