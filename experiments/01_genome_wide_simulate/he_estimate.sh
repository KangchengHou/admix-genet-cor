#!/bin/bash -l
#$ -cwd
#$ -l h_data=16G,h_rt=0:30:00,highp
#$ -j y
#$ -o ./job_out

# bash he_estimate.sh chr1chr2_20_80 indiv_25000_snp_115651 
. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/anaconda3/bin:$PATH
export PYTHONNOUSERSITE=True

geno_id=$1
subset_prefix=$2
param_prefix=$3

phen=02_phe/${geno_id}/${subset_prefix}/${param_prefix}.phen
mkdir -p 03_estimate/${geno_id}/${subset_prefix}/he_estimate
grm_dir=03_estimate/${geno_id}/${subset_prefix}/grm

python he_estimate.py \
    --grm_dir ${grm_dir} \
    --phen ${phen} \
    --out 03_estimate/${geno_id}/${subset_prefix}/he_estimate/${param_prefix}.csv