#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=16G,h_rt=0:30:00
#$ -j y
#$ -o ./job_out

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

trait=$1

plink2 --vcf /u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s03_aframr/dataset/hm3.vcf.gz \
    --pheno out/plink_assoc/pheno.${trait}.txt \
    --covar out/plink_assoc/covar.${trait}.txt \
    --linear hide-covar \
    --covar-variance-standardize \
    --out out/plink_assoc/assoc.${trait}

mv out/plink_assoc/assoc.${trait}.${trait}.glm.linear out/plink_assoc/assoc.${trait}.glm.linear
