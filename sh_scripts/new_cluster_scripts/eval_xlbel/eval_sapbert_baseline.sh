#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_xlbel/eval_xlbel_sapbert_baseline.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_xlbel/eval_xlbel_sapbert_baseline.txt       # Файл для вывода результатов
#SBATCH --time=23:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1


echo "model: Sapbert, DATA: all"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/all_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/all" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: en"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/en_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/en" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "model: Sapbert, DATA: fi"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/fi_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/fi" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "model: Sapbert, DATA: ko"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ko_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/ko" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "model: Sapbert, DATA: th"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/th_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/th" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: zh"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/zh_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/zh" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: de"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/de_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/de" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: es"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/es_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/es" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: ja"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ja_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/ja" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: ru"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ru_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/ru" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model: Sapbert, DATA: tr"
python eval.py \
	--model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/tr_1k_test_query.txt" \
	--output_dir "/home/etutubalina/graph_entity_linking/evaluation_xlbel/sapbert_baseline/tr" \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls
