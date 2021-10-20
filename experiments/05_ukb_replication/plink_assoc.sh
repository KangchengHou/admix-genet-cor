#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=16G,h_rt=0:20:00
#$ -j y
#$ -o ./job_out

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

OLD_DIR=/u/project/pasaniuc/kangchen/tractor-response/ukb

trait=$1
chrom=1

OUT_DIR=out/plink_assoc/
plink2 --vcf ${OLD_DIR}/02_impute/processed/chr${chrom}/chr${chrom}.sample.typed.vcf.gz \
    --pheno out/data/${trait}.pheno \
    --covar out/data/covar.txt \
    --linear hide-covar \
    --covar-variance-standardize \
    --out out/plink_assoc/assoc.${trait}.${chrom}

mv ${OUT_DIR}/assoc.${trait}.${chrom}.${trait}.glm.linear ${OUT_DIR}/assoc.${trait}.${chrom}.glm.linear