geno=./01_data/01_simple/admix.zarr
out_dir=02_simulate/01_simple/

mkdir -p ${out_dir}

for gamma in 0.0 0.05 0.08 0.1; do
echo $gamma
python simulate.py \
    --geno ${geno} \
    --var_g 0.1 --var_e 1.0 --gamma ${gamma} \
    --out_prefix ${out_dir}/gamma_${gamma}
done
