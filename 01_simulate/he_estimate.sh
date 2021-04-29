geno_id=$1
param_prefix=$2


phen=02_phe/${geno_id}/${param_prefix}.phen
mkdir -p 03_estimate/${geno_id}/he_estimate
grm_dir=03_estimate/${geno_id}/grm

python he_estimate.py \
    --grm_dir ${grm_dir} \
    --phen ${phen} \
    --out 03_estimate/${geno_id}/he_estimate/${param_prefix}.csv