id=$1
geno=$(cat config.yaml | shyaml get-value experiments.${id}.geno)

out_dir=03_estimate/${id}/

mkdir -p ${out_dir}/grm

python gcta_estimate.py \
    cli_build_grm \
    --geno ${geno} \
    --out_prefix ${out_dir}/grm/
