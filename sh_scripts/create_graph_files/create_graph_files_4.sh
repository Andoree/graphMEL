#!/bin/bash
#SBATCH --job-name=create_umls_graph_files          # Название задачи
#SBATCH --error=../logs/create_umls_graph_files.err        # Файл для вывода ошибок
#SBATCH --output=../logs/create_umls_graph_files.txt       # Файл для вывода результатов
#SBATCH --time=07:50:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../scripts/preprocessing/reformat_umls_to_graph.py --mrconso "../UMLS/2020AB/MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_ENG_FRE_GER_SPA_DUT_MRCONSO_filt.RRF" \
--mrrel "../UMLS/2020AB/MRREL.RRF" \
--output_node_id2synonyms_path "../data/umls_graph/2020AB/FULL_MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_ENG_FRE_GER_SPA_DUT_split/synonyms" \
--output_node_id2cui_path "../data/umls_graph/2020AB/FULL_MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_ENG_FRE_GER_SPA_DUT_split/id2cui" \
--output_edges_path "../data/umls_graph/2020AB/FULL_MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_ENG_FRE_GER_SPA_DUT_split/edges"

