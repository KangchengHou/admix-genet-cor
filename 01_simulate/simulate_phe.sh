id=$1
geno=$(cat config.yaml | shyaml get-value experiments.${id}.geno)
echo $geno

out_dir=02_phe/${id}
mkdir -p ${out_dir}

for gamma in 0.0 0.05 0.08 0.1; do
echo $gamma
python simulate_phe.py \
    --geno ${geno} \
    --var_g 0.1 --var_e 1.0 --gamma ${gamma} \
    --out_prefix ${out_dir}/gamma_${gamma}
done
