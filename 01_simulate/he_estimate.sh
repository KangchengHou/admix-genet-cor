dataset_prefix=01_simple

mkdir -p 03_estimate/${dataset_prefix}/he_estimate

for gamma in 0.0 0.05 0.08 0.1; do
    param_prefix=gamma_${gamma}

    grm_dir=03_estimate/${dataset_prefix}/grm
    phen=02_simulate/${dataset_prefix}/${param_prefix}.phen
    python estimate_he.py \
        --grm_dir ${grm_dir} \
        --phen ${phen} \
        --out 03_estimate/${dataset_prefix}/he_estimate/${param_prefix}.csv    
done