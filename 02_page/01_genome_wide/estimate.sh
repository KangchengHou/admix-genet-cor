#!/bin/bash -l
#$ -cwd
#$ -l h_data=60G,h_rt=0:30:00
#$ -j y
#$ -o ./job_out
#$ -t 1-24
. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

trait_i=$((SGE_TASK_ID - 1))

python estimate.py --trait_i ${trait_i} --out_dir out/estimate/
