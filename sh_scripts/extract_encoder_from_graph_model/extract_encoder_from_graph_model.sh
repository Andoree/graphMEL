#!/bin/bash
#SBATCH --job-name=extract_encoder_from_graph_model          # Название задачи
#SBATCH --error=../../logs/extract_encoder_from_graph_model.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/extract_encoder_from_graph_model.txt       # Файл для вывода результатов
#SBATCH --time=00:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1

python ../../scripts/postprocessing/extract_encoder_from_graph_model.py --model_checkpoint_path="../../pretrained_models/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-3.0-2.0-5_b128_lr2e-05/checkpoint_e_19_steps_20360.pth" \
--graph_model_dir="../../pretrained_models/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-3.0-2.0-5_b128_lr2e-05" \
--output_dir="../../pretrained_encoders/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-3.0-2.0-5_b128_lr2e-05_checkpoint_e_19_steps_20360/"

