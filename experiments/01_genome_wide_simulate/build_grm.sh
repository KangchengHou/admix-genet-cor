#!/bin/bash -l
#$ -cwd
#$ -l h_data=28G,h_rt=2:30:00,highp
#$ -j y
#$ -o ./job_out

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

id=$1
indiv_subset=$2
snp_subset=$3
# bash build_grm.sh twochr_20_80 indiv_5000 snp_12500
geno=$(cat config.yaml | shyaml get-value experiments.${id}.geno)
out_dir=03_estimate/${id}/${indiv_subset}_${snp_subset}

mkdir -p ${out_dir}/grm

python gcta_estimate.py \
    cli_build_grm \
    --geno ${geno} \
    --indiv_subset 00_meta/subsets/${indiv_subset}.txt \
    --snp_subset 00_meta/subsets/${snp_subset}.txt \
    --out_prefix ${out_dir}/grm/
