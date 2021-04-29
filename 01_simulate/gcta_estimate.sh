#!/bin/bash -l
#$ -cwd
#$ -l h_data=10G,h_rt=0:30:00,highp
#$ -j y
#$ -o ./job_out
#$ -t 1-100

geno_id=$1
param_prefix=$2

i_sim=${SGE_TASK_ID}

pheno_dir=02_phe/${geno_id}/
grm_dir=03_estimate/${geno_id}/grm
out_dir=03_estimate/${geno_id}/gcta_estimate/${param_prefix}

mkdir -p ${out_dir}

gcta_path=../gcta64

./${gcta_path} --reml --mgrm ${grm_dir}/mgrm.txt \
    --pheno ${pheno_dir}/${param_prefix}.phen \
    --mpheno ${i_sim} \
    --out ${out_dir}/${i_sim} \
    --reml-no-lrt \
    --reml-no-constrain
