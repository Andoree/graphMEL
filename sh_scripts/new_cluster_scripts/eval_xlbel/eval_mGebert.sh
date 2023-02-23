#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_xlbel/eval_xlbel_mGEBERT.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_xlbel/eval_xlbel_mGEBERT.txt       # Файл для вывода результатов
#SBATCH --time=23:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1


echo "cmodel: mGebert, DATA: all"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/all_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/all" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: en"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/en_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/en" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "cmodel: mGebert, DATA: fi"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/fi_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/fi" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "cmodel: mGebert, DATA: ko"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ko_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/ko" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "cmodel: mGebert, DATA: th"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/th_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/th" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: zh"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/zh_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/zh" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: de"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/de_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/de" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: es"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/es_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/es" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: ja"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ja_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/ja" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: ru"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ru_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/ru" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "cmodel: mGebert, DATA: tr"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT//" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/tr_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/mGebert/tr" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls
