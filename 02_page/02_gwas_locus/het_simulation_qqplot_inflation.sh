#!/bin/bash -l
#$ -cwd
#$ -l h_data=16G,h_rt=0:30:00
#$ -j y
#$ -o ./job_out
#$ -t 1-50
. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

region_i=$((SGE_TASK_ID - 1))

python het_simulation_qqplot_inflation.py --region_i ${region_i} --out_dir out/het_simulation/