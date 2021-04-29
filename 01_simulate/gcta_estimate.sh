#!/bin/bash -l
#$ -cwd
#$ -l h_data=4G,h_rt=0:10:00,highp
#$ -j y
#$ -o ./job_out
#$ -t 1-100

param_prefix=$1
i_sim=${SGE_TASK_ID}
dataset_prefix=01_simple
# param_prefix=gamma_0.1

out_dir=./03_estimate/${dataset_prefix}/gcta_estimate/${param_prefix}
grm_dir=./03_estimate/${dataset_prefix}/grm
pheno_dir=02_simulate/${dataset_prefix}/
mkdir ${out_dir}

gcta_path=../gcta64

./${gcta_path} --reml --mgrm ${grm_dir}/mgrm.txt \
    --pheno ${pheno_dir}/${param_prefix}.phen \
    --mpheno ${i_sim} \
    --out ${out_dir}/${i_sim} \
    --reml-no-lrt \
    --reml-no-constrain
