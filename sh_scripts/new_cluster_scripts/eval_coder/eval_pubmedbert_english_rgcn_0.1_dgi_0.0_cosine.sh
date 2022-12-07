#!/bin/bash
#SBATCH --job-name=ev_coder          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/CODER/test/eval_results/pubmedbert_english_rgcn_0.1_dgi_0.0/cosine_0.1.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/CODER/test/eval_results/pubmedbert_english_rgcn_0.1_dgi_0.0/cosine_0.1.txt       # Файл для вывода результатов
#SBATCH --time=18:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a


python /home/echernyak/graph_entity_linking/CODER/test/cadec/cadec_eval.py "/home/echernyak/graph_entity_linking/results/pretrained_encoders/2020AB/DGI_GRAPH_LOSS_RGCN/ENGLISH_FULL_PUBMEDBERT/dgi_0.0_rgcn_1_3_[3]_text_1.0_remove_loops_Truegraph_loss_0.1_intermodal_cosine_0.1_0.3_256--None-64_rel_intermodal_miner_True_lr_2e-05_b_128_fast_rgcn_conv/checkpoint_e_1_steps_94765.pth/"

