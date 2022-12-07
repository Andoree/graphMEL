#!/bin/bash
#SBATCH --job-name=ev_coder          # Название задачи
#SBATCH --error=/home/echernyak/graph_entity_linking/CODER/test/eval_results/pubmedbert_english_gatv2_0.1_dgi_0.0/sapbert_0.1.err        # Файл для вывода ошибок
#SBATCH --output=/home/echernyak/graph_entity_linking/CODER/test/eval_results/pubmedbert_english_gatv2_0.1_dgi_0.0/sapbert_0.1.txt       # Файл для вывода результатов
#SBATCH --time=18:59:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a


python /home/echernyak/graph_entity_linking/CODER/test/cadec/cadec_eval.py "/home/echernyak/graph_entity_linking/results/pretrained_encoders/2020AB/DGI_GRAPH_LOSS_GatV2/ENGLISH_FULL_PUBMEDBERT/gatv2_3_256_1_3_0.3_2_0.1_graph_loss_0.1_rel_remove_loops_True_dgi_0.0_text_loss_1.0_intermodal_sapbert_intermodal_miner_True_relational_features_True_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/" 

