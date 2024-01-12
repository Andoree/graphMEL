#!/bin/bash
#SBATCH --job-name=ev_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_eval/biosyn_medical_crossing_TAC2017ADR.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_eval/biosyn_medical_crossing_TAC2017ADR.txt       # Файл для вывода результатов
#SBATCH --time=22:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1



TAC2017ADR_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"
TAC2017ADR_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test"
TAC2017ADR_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test_refined"

echo "Evaluating ensapbert..."
echo "Evaluating ensapbert..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/ensapbert/tac2017adr/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/ensapbert/tac2017adr/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path /home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/TAC_ADR/ \
    --dictionary_path $TAC2017ADR_TEST_DICT \
    --data_dir $TAC2017ADR_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/ensapbert/tac2017adr/full_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path /home/etutubalina/graph_entity_linking/results/trained_biosyn_models/ensapbert/TAC_ADR/ \
    --dictionary_path $TAC2017ADR_TEST_DICT \
    --data_dir $TAC2017ADR_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/ensapbert/tac2017adr/refined_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions


echo "Evaluating encoder..."
echo "Evaluating encoder..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/encoder/tac2017adr/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/encoder/tac2017adr/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path /home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/TAC_ADR/ \
    --dictionary_path $TAC2017ADR_TEST_DICT \
    --data_dir $TAC2017ADR_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/encoder/tac2017adr/full_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path /home/etutubalina/graph_entity_linking/results/trained_biosyn_models/encoder/TAC_ADR/ \
    --dictionary_path $TAC2017ADR_TEST_DICT \
    --data_dir $TAC2017ADR_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/encoder/tac2017adr/refined_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions



