#!/bin/bash -l
#$ -cwd
#$ -l h_data=10G,h_rt=0:30:00,highp
#$ -j y
#$ -o ./job_out


n_indiv=$1
# ancestry proportion for the first ancestry
anc_prop=$2

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

other_anc_prop=$(bc <<< "scale=1;1.0 - ${anc_prop}")
echo "[${anc_prop},${other_anc_prop}]"
python utils.py prepare_dataset \
    --n_indiv ${n_indiv} --n_snp 2000 \
    --anc_props "[${anc_prop},${other_anc_prop}]" \
    --out out/dataset/n_indiv_${n_indiv}_anc_prop_${anc_prop}.zarr