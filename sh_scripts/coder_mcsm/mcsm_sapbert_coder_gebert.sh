#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/mcsm/mcsm_sapbert_coder_gebert.err         # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/mcsm/mcsm_sapbert_coder_gebert.txt       # Файл для вывода результатов
#SBATCH --time=10:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1


python /home/etutubalina/graph_entity_linking/CODER/test/embeddings_reimplement/mcsm.py \
      --umls_dir "/home/etutubalina/graph_entity_linking/UMLS/2020AB/" \
      --bert_encoder_paths "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" \
      "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_all" \
      "/home/etutubalina/graph_entity_linking/huggingface_models/GEBERT/mGEBERT"

