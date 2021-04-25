geno_prefix=./01_data/01_simple/admix
out_dir=03_estimate/01_simple/

mkdir -p ${out_dir}/grm

python gcta_estimate.py \
    --cli_build_grm \
    --geno_prefix ${geno_prefix} \
    --out_prefix ${out_dir}/grm/
