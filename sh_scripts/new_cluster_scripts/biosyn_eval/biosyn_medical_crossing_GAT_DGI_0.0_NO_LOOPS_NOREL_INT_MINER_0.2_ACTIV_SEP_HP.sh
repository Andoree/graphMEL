#!/bin/bash
#SBATCH --job-name=ev_biosyn          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_eval/eval_biosyn_GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/biosyn_eval/eval_biosyn_GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP.txt       # Файл для вывода результатов
#SBATCH --time=22:58:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a
#SBATCH --nodes=1



NCBID_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/test_dictionary.txt"
NCBID_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_test"
NCBID_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/ncbi/processed_test_refined"

BC5CDRD_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/test_dictionary.txt"
BC5CDRD_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_test"
BC5CDRD_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-disease/processed_test_refined"


BC5CDRC_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/test_dictionary.txt"
BC5CDRC_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_test"
BC5CDRC_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc5cdr-chemical/processed_test_refined"


BC2GN_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/test_dictionary.txt"
BC2GN_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_test"
BC2GN_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/bc2gm/processed_test_refined"

SMM4H_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/test_dictionary.txt"
SMM4H_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/processed_test"
SMM4H_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/smm4h/processed_test_refined"


TAC2017ADR_TEST_DICT="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"
TAC2017ADR_TEST_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test"
TAC2017ADR_TEST_REFINED_DIR="/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test_refined"


echo "Evaluating NCBI..."
echo "Evaluating NCBI..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/" \
    --dictionary_path $NCBID_TEST_DICT \
    --data_dir $NCBID_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/full_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

echo "Evaluating NCBI refined..."
echo "Evaluating NCBI refined..." 1>&2
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/" \
    --dictionary_path $NCBID_TEST_DICT \
    --data_dir $NCBID_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/NCBI/refined_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions


echo "Evaluating BC5CDRD..."
echo "Evaluating BC5CDRD..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/" \
    --dictionary_path $BC5CDRD_TEST_DICT \
    --data_dir $BC5CDRD_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

echo "Evaluating BC5CDRD refined..."
echo "Evaluating BC5CDRD refined..." 1>&2
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/" \
    --dictionary_path $BC5CDRD_TEST_DICT \
    --data_dir $BC5CDRD_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRD/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions


echo "Evaluating BC5CDRC..."
echo "Evaluating BC5CDRC..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/" \
    --dictionary_path $BC5CDRC_TEST_DICT \
    --data_dir $BC5CDRC_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

echo "Evaluating BC5CDRC refined..."
echo "Evaluating BC5CDRC refined..." 1>&2
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/" \
    --dictionary_path $BC5CDRC_TEST_DICT \
    --data_dir $BC5CDRC_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC5CDRC/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions


echo "Evaluating BC2GN..."
echo "Evaluating BC2GN..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/" \
    --dictionary_path $BC2GN_TEST_DICT \
    --data_dir $BC2GN_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

echo "Evaluating BC2GN refined..."
echo "Evaluating BC2GN refined..." 1>&2
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/" \
    --dictionary_path $BC2GN_TEST_DICT \
    --data_dir $BC2GN_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/BC2GN/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions


echo "Evaluating SMM4H..."
echo "Evaluating SMM4H..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/SMM4H/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/SMM4H/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/SMM4H/" \
    --dictionary_path $SMM4H_TEST_DICT \
    --data_dir $SMM4H_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/SMM4H/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

echo "Evaluating SMM4H refined..."
echo "Evaluating SMM4H refined..." 1>&2
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path "/home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/SMM4H/" \
    --dictionary_path $SMM4H_TEST_DICT \
    --data_dir $SMM4H_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/SMM4H/" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

echo "Evaluating TAC_ADR refined..."
echo "Evaluating TAC_ADR refined..." 1>&2
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/tac2017adr/full_test"
mkdir -p "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/tac2017adr/refined_test"
python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path /home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/TAC2017ADR/ \
    --dictionary_path $TAC2017ADR_TEST_DICT \
    --data_dir $TAC2017ADR_TEST_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/tac2017adr/full_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions

python /home/etutubalina/graph_entity_linking/BioSyn/eval.py \
    --model_name_or_path /home/etutubalina/graph_entity_linking/results/trained_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/TAC2017ADR/ \
    --dictionary_path $TAC2017ADR_TEST_DICT \
    --data_dir $TAC2017ADR_TEST_REFINED_DIR \
    --output_dir "/home/etutubalina/graph_entity_linking/results/evaluation_biosyn_models/GAT_DGI_0.0_NO_LOOPS_NOREL_INT_MINER_0.2_ACTIV_SEP_HP/tac2017adr/refined_test" \
    --use_cuda \
    --topk 20 \
    --max_length 25 \
    --save_predictions



