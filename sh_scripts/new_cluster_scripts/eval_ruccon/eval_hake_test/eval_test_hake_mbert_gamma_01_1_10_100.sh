#!/bin/bash
#SBATCH --job-name=eval_bert_ranking          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_ruccon/eval_test_hake_mbert_gamma_01_1_10_100.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/graphmel/logs/eval_ruccon/eval_test_hake_mbert_gamma_01_1_10_100.txt       # Файл для вывода результатов
#SBATCH --time=04:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1


echo "GAMMA 0.1 TEST FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_0.1_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-full" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

echo " GAMMA 0.1 TEST ZERO SHOT"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_0.1_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-zero_shot" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

echo "GAMMA 1.0 TEST FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_1.0_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-full" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

echo "GAMMA 1.0 TEST ZERO SHOT"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_1.0_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-zero_shot" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"


echo "GAMMA 10.0 TEST FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_10.0_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-full" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

echo "GAMMA 10.0 TEST ZERO SHOT"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_10.0_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-zero_shot" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"


echo "GAMMA 100.0 TEST FULL"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_100.0_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-full" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

echo "GAMMA 100.0 TEST ZERO SHOT"
python /home/etutubalina/graph_entity_linking/Fair-Evaluation-BERT/eval_bert_ranking.py \
--model_dir "/home/etutubalina/graph_entity_linking/results/pretrained_encoders/2020AB/HAKE_SAPBERT/RU_TEST_MBERT_0.1/neg_4_gamma_100.0_mw_1.0_pw_0.0_adv_1.0_filt_trans_False_filt_sem_type_False_hake_weight_0.1_lr_2e-05_b_128/checkpoint_e_1_steps_2770.pth" \
--data_folder "/home/etutubalina/graph_entity_linking/RuCCoN/final/test-zero_shot" \
--vocab "/home/etutubalina/graph_entity_linking/RuCCoN/umls_rus_biosyn_fmt.txt"

