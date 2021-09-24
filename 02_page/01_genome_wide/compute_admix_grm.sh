#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=120G,h_rt=4:00:00
#$ -j y
#$ -o ./job_out
#$ -t 1-22
. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

chrom=${SGE_TASK_ID}
python compute_admix_grm.py --chrom ${chrom}
