
# Estimation with GREML
```bash
for geno_id in simple chr1chr2_20_80 chr1chr2_50_50; do
    for gamma in 0.0 0.05 0.08 0.1; do
        qsub gcta_estimate.sh ${geno_id} gamma_${gamma}
    done
done
```


# Hypothesis testing with GREML
```bash
# for geno_id in simple chr1chr2_20_80 chr1chr2_50_50; do
for geno_id in simple; do
    for gamma in 0.0 0.05 0.08 0.1; do
        qsub gcta_estimate_reduced.sh ${geno_id} gamma_${gamma}
    done
done
```

# Estimation with HE regression
```bash
for geno_id in simple chr1chr2_20_80 chr1chr2_50_50; do
    for gamma in 0.0 0.05 0.08 0.1; do
        bash he_estimate.sh ${geno_id} gamma_${gamma}
    done
done
```


