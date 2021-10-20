#!/bin/bash -l
#$ -cwd
#$ -l h_data=40G,h_rt=0:30:00,highp
#$ -j y
#$ -o ./job_out

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/anaconda3/bin:$PATH
export PYTHONNOUSERSITE=True

id=$1
indiv_subset=$2
snp_subset=$3

geno=$(cat config.yaml | shyaml get-value experiments.${id}.geno)
echo $geno

out_dir=02_phe/${id}/${indiv_subset}_${snp_subset}
mkdir -p ${out_dir}

for gamma in 0.8 1.0; do
    echo $gamma
    python simulate_phe.py \
        --geno ${geno} \
        --var_g 1.0 --var_e 1.0 --gamma ${gamma} \
        --indiv_subset 00_meta/subsets/${indiv_subset}.txt \
        --snp_subset 00_meta/subsets/${snp_subset}.txt \
        --out_prefix ${out_dir}/gamma_${gamma}
done
