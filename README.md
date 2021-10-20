TODO:
1. Simulate from Mexican data, use the inferred local ancestry as groundtruth.
2. Cope with 3-way admixture.

`gcta_nr_robust` from https://github.com/gusevlab/fusion_twas/blob/master/gcta_nr_robust


# Experiments to replicate the results
1. Simulation on chromsome 22 (typed) with (a) different genetic architecture (b) 

python simulate.py --pfile XX --lanc XX --h2 XX --gamma XX --out_dir XX 

2. Estimation process

python estimate.py --pfile XX --lanc XX --pheno XX --out_dir XX


3. Real data analysis 

python estimate.py --pfile XX --lanc XX --pheno XX --out_dir XX
