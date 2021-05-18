```bash
# LIST_GENO_ID="simple"
LIST_GENO_ID="twochr_20_80 twochr_50_50"
```
# Step 1: simulate phenotype
```bash
for geno_id in $LIST_GENO_ID; do
    indiv_subset="indiv_25000"
    for snp_subset in snp_28912 snp_57825 snp_115651; do
        qsub simulate_phe.sh ${geno_id} ${indiv_subset} ${snp_subset}
    done
done
```

# Step 2: calculate GRM
```bash
for geno_id in $LIST_GENO_ID; do
    snp_subset="snp_50000"
    for indiv_subset in indiv_5000 indiv_10000; do
        qsub build_grm.sh ${geno_id} ${indiv_subset} ${snp_subset}
    done

    indiv_subset="indiv_25000"
    for snp_subset in snp_12500 snp_25000 snp_50000; do
        qsub build_grm.sh ${geno_id} ${indiv_subset} ${snp_subset}
    done
done
```

# Step 3: Estimation with HE regression

```bash
for geno_id in $LIST_GENO_ID; do
    indiv_subset="indiv_25000"
    for snp_subset in snp_28912 snp_57825 snp_115651; do
        subset_prefix=${indiv_subset}_${snp_subset}
        for gamma in 0.8 1.0; do
            qsub he_estimate.sh ${geno_id} ${subset_prefix} gamma_${gamma}
        done
    done
done
```

# Step 4: Estimation with GREML
```bash
for geno_id in $LIST_GENO_ID; do
    for gamma in 0.8 1.0; do
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




