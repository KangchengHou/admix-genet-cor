#!/bin/bash -l
#$ -cwd
#$ -l h_data=10G,h_rt=0:30:00,highp
#$ -j y
#$ -o ./job_out
#$ -t 1-100

# Estimate reduced model
geno_id=$1
param_prefix=$2

i_sim=${SGE_TASK_ID}

pheno_dir=02_phe/${geno_id}/
grm_dir=03_estimate/${geno_id}/grm
out_dir=03_estimate/${geno_id}/gcta_estimate_reduced/${param_prefix}

mkdir -p ${out_dir}

gcta_path=../gcta64

./${gcta_path} --reml --grm ${grm_dir}/K_full \
    --pheno ${pheno_dir}/${param_prefix}.phen \
    --mpheno ${i_sim} \
    --out ${out_dir}/${i_sim} \
    --reml-no-lrt \
    --reml-no-constrain
