#!/bin/bash
#SBATCH --job-name=extract_encoder_from_graph_model          # Название задачи
#SBATCH --error=../../logs/extract_encoder_from_graph_model.err        # Файл для вывода ошибок
#SBATCH --output=../../logs/extract_encoder_from_graph_model.txt       # Файл для вывода результатов
#SBATCH --time=00:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу

python ../../scripts/postprocessing/extract_encoder_from_graph_model.py --graph_network_architecture="node2vec" \
--model_checkpoint_path="../../pretrained_models/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-3.0-2.0-5_b160_lr2e-05/checkpoint_e_5_steps_4884.pth" \
--graph_model_dir="../../pretrained_models/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-3.0-2.0-5_b160_lr2e-05/" \
--output_dir="../../pretrained_encoders/2020AB/Node2vec/RU_split/nns-c-wpn-p-q-wl_1-1-3.0-2.0-5_b160_lr2e-05/"

