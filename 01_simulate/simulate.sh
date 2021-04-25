geno_prefix=./01_data/01_simple/admix
out_dir=02_simulate/01_simple/

mkdir -p ${out_dir}

for gamma in 0.0 0.05 0.08 0.1; do
python simulate.py \
    --geno_prefix ${geno_prefix} \
    --var_g 0.1 --var_e 1.0 --gamma ${gamma} \
    --out_prefix ${out_dir}/gamma_${gamma}
done
