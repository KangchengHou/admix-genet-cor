#!/bin/bash -l
#$ -cwd
#$ -l h_data=32G,h_rt=0:60:00
#$ -j y
#$ -o ./job_out
#$ -t 1-50

dset_prefix=$1
cor=$2

i_sim=$((SGE_TASK_ID - 1))

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

mkdir -p out/estimate/${dset_prefix}/cor_${cor}

python utils.py estimate \
    --dset out/dataset/${dset_prefix}.zarr \
    --pheno out/pheno/${dset_prefix}/cor_${cor}.npy \
    --method HE \
    --i_sim ${i_sim} \
    --out out/estimate/${dset_prefix}/cor_${cor}/HE.${i_sim}.txt
