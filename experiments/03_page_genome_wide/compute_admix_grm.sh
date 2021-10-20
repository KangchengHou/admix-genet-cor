#!/bin/bash -l
#$ -cwd
#$ -l h_data=50G,h_rt=1:00:00,highp
#$ -j y
#$ -o ./job_out
#$ -t 1-22
. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

chrom=${SGE_TASK_ID}
python compute_admix_grm.py --chrom ${chrom}
