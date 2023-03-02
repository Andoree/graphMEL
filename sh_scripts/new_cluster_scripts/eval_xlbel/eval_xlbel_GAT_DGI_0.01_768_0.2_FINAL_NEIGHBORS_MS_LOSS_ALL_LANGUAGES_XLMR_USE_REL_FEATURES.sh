#!/bin/bash
#SBATCH --job-name=eval_xlbel          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_xlbel/GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_USE_REL_FEATURES.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_xlbel/GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_USE_REL_FEATURES.txt       # Файл для вывода результатов
#SBATCH --time=23:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1

MODEL="/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES/GAT_DGI_0.0_MULTILINGUAL_XLMR/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_remove_loops_False_dgi_0.01_text_loss_1.0_intermodal_sapbert_intermodal_miner_True_0.2_relational_features_freeze_neigh_False_True_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_172907.pth/"
MODEL_DESC="GAT_DGI_0.01_768_0.2_FINAL_NEIGHBORS_MS_LOSS_ALL_LANGUAGES_XLMR_USE_REL_FEATURES"


mkdir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}
echo "model : '${MODEL_DESC}', DATA: all"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/all_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/all \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: en"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/en_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/en \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "model : '${MODEL_DESC}', DATA: fi"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/fi_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/fi \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "model : '${MODEL_DESC}', DATA: ko"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ko_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/ko \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls


echo "model : '${MODEL_DESC}', DATA: th"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/th_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/th \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: zh"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/zh_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/zh \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: de"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/de_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/de \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: es"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/es_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/es \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: ja"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ja_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/ja \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: ru"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/ru_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/ru \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls

echo "model : '${MODEL_DESC}', DATA: tr"
python eval.py \
	--model_dir $MODEL \
	--dictionary_path "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/umls_onto_all_lang_cased_wikimed_only_399931.txt" \
	--data_dir "/home/etutubalina/graph_entity_linking/sapbert/evaluation/xl_bel/xlbel_v0.0/tr_1k_test_query.txt" \
	--output_dir /home/etutubalina/graph_entity_linking/evaluation_xlbel/${MODEL_DESC}/tr \
	--use_cuda \
	--max_length 25 \
	--custom_query_loader \
	--agg_mode cls
