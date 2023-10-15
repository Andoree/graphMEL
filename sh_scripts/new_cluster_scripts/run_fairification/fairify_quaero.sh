#!/bin/bash
#SBATCH --job-name=fairify          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fairification/fairify_quaero.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fairification/fairify_quaero.txt       # Файл для вывода результатов
#SBATCH --time=05:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

THRESHOLD=0.2

for SUBSET in "BOTH" "EMEA" "MEDLINE"; do

  mkdir /home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_filtered_${THRESHOLD}/${SUBSET}
  python3 /home/etutubalina/graph_entity_linking/medical_crossing/fairification.py \
                  --test_dir /home/etutubalina/graph_entity_linking/data_medical_crossing/datasets/QUAERO_full_biosyn_format/${SUBSET} \
                  --vocabulary /home/etutubalina/graph_entity_linking/data_medical_crossing/vocabs/mantra_fr_dict_ALL.txt
                  --levenshtein_norm_method 1 \
                  --levenshtein_threshold $THRESHOLD

done






