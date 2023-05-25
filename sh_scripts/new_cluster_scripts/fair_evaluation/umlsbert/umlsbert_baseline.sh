#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/umlsbert_baseline.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/umlsbert_baseline.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=5                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/UMLSBert/umlsbert/"


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


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test"
echo VOCAB "codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_exact"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_exact" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_exact_vocab"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/DIAGNOSTICO/ SPLIT test-fair_levenshtein_train_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/DIAGNOSTICO/test-fair_levenshtein_train_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-d-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test"
echo VOCAB "codiesp-p-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_exact"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_exact" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_exact_vocab"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset codiesp/PROCEDIMIENTO/ SPLIT test-fair_levenshtein_train_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/codiesp/PROCEDIMIENTO/test-fair_levenshtein_train_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/codiesp-p-codes-es.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts"
echo VOCAB "CANTEMIST-lopez-ubeda-et-al.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_exact"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_exact" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_exact_vocab"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_exact_vocab" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_levenshtein_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_levenshtein_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


echo "dataset cantemist/test-set/ SPLIT cantemist-norm-concepts-fair_levenshtein_train_0.2"
echo VOCAB "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir $MODEL \
    --data_folder "/home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/cantemist/test-set/cantemist-norm-concepts-fair_levenshtein_train_0.2" \
    --vocab "/home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/CANTEMIST-lopez-ubeda-et-al.txt"


