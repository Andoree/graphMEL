#!/bin/bash
#SBATCH --job-name=fair_ev          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_tac_adr.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/fair_evaluation/zeroshot_medical_crossing_tac_adr.txt       # Файл для вывода результатов
#SBATCH --time=20:30:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --gpus=1


export CUDA_VISIBLE_DEVICES=0
MODEL="/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/"

echo "ensapbert"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"


echo "encoder"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "/home/etutubalina/graph_entity_linking/huggingface_models/GanjinZero/coder_eng/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"

echo "engebert_graphsage_neighbors_loss_3_dgi_0.01_b128_NO_N_MS"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH/GRAPHSAGE_DGI_MULTILINGUAL/gs_1-3_text_loss_1.0_768_3_0.3_remove_loops_True_graph_0.1_dgi_0.01_modal_sapbert_0.1_intermodal_miner_True_0.2_text_loss_neighbors_False_freeze_neigh_False_lr_2e-05_b_128/checkpoint_e_1_steps_94765.pth/"  \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"

echo "GAT_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH_NEW/GAT_DGI/gatv2_3_768_1_3_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.01_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_hard_det_txt_False_0.1_lr_2e-05_b_128_ACTIV/checkpoint_e_1_steps_94765.pth/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"

echo "GAT_NO_ACTIV_NOREL_HARD_NODETACH_DGI_0.01_INT_0.1_GRAPH_0.1_b128"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/768_0.2_ENGLISH_NEW/GAT_DGI/gatv2_3_768_1_2_0.3_2_0.1_graph_loss_0.1_rel_NEW_rl_True_dgi_0.01_tl_1.0_inter_sapbert_intermodal_miner_True_0.2_rel_feat_False_freeze_neigh_False_tl_neighbors_False_ilt_sapbert_istrat_hard_det_txt_False_0.1_lr_2e-05_b_128_NO_ACTIV/checkpoint_e_1_steps_94765.pth/" \
    --data_folder "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/processed_test" \
    --vocab "/home/etutubalina/graph_entity_linking/fair_eval_coling_datasets/tac2017adr/test_dictionary.txt"




