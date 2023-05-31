#!/bin/bash
#SBATCH --job-name=create_pos_pairs          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/create_pos_pairs_dataset/create_pos_pairs_dataset_full_multilingual_all_languages_stratified_fixed_limit_eng_20_50.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/create_pos_pairs_dataset/create_pos_pairs_dataset_full_multilingual_all_languages_stratified_fixed_limit_eng_20_50.txt       # Файл для вывода результатов
#SBATCH --time=19:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --constraint=type_b

python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/preprocessing/create_positive_triplets_dataset_stratified.py --mrconso "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRCONSO.RRF" \
--mrrel "/home/etutubalina/graph_entity_linking/UMLS/2020AB/MRREL.RRF" \
--langs "ENG" "SPA" "POR" "FRE" "JPN" "RUS" "DUT" "GER" "ITA" "CZE" "SWE" "KOR" "LAV" "HUN" "CHI" "NOR" "POL" "TUR" "EST" "FIN" "SCR" "UKR" "GRE" "DAN" "BAQ" "HEB" \
--max_pairs_per_single_lang 20 \
--max_pairs_crosslingual 50 \
--limit_english_only \
--output_dir "/home/etutubalina/graph_entity_linking/pos_pairs_graph_data/2020AB/MULTILINGUAL_ALL_LANGUAGES_FIXED_LIMIT_ENG_20_50_FULL/"

