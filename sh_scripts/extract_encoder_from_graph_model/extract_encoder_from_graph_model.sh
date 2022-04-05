#!/bin/bash
#SBATCH --job-name=extract_encoder_from_graph_model          # Название задачи
#SBATCH --error=../../logs/extract_encoder_from_graph_model/extract_encoder_from_graph_model.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/extract_encoder_from_graph_model/extract_encoder_from_graph_model.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python ../../scripts/postprocessing/extract_encoder_from_graph_model.py --input_pretrained_graph_models_dir="../../pretrained_models/2020AB/Node2vec/RU_split/" \
--bert_initialization_model="../../models/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR/" \
--output_dir="../../pretrained_encoders/2020AB/Node2vec/RU_split/"