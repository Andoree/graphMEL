#!/bin/bash
#SBATCH --job-name=count_concept_defintions          # Название задачи
#SBATCH --error=delete.err        # Файл для вывода ошибок
#SBATCH --output=delete.txt       # Файл для вывода результатов
#SBATCH --time=00:20:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу

python ../scripts/statistics/count_concept_descriptions.py --mrconso "../UMLS/2020AB/MSH_MDR_SNOMEDCT_US_ICD10CM_ICD9CM_ICD10_DRUGBANK_RXNORM_ENG_FRE_GER_SPA_DUT_MRCONSO_filt.RRF" \
--groupby_sab \
--mrdef "../UMLS/2020AB/MRDEF.RRF" \
--groupby_stats_output_path "../statistics/stats_bysab_2020AB.tsv" \
--global_stats_output_path "../statistics/stats_2020AB.txt"
