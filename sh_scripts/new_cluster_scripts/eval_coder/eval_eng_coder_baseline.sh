#!/bin/bash
#SBATCH --job-name=ev_coder          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/CODER/test/eval_results/baselines/eng_coder_baseline.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/CODER/test/eval_results/baselines/eng_coder_baseline.txt       # Файл для вывода результатов
#SBATCH --time=18:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a


python /home/echernyak/graph_entity_linking/CODER/test/cadec/cadec_eval.py "/home/echernyak/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/"

