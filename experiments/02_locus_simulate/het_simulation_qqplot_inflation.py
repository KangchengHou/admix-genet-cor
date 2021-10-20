import numpy as np
import pandas as pd
import pickle
import xarray as xr
import admix
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from os.path import join
import sys
import fire

sys.path.append("../../")
import common

np.random.seed(1234)


def run(region_i, out_dir):
    # read dataset and do minimal processing
    dset_gwas_hit = xr.open_zarr("out/locus_het/gwas_hit.zarr/", chunks=-1)
    admix.tools.af_per_anc(dset_gwas_hit)
    maf = np.minimum(
        dset_gwas_hit["af_per_anc"].values, 1 - dset_gwas_hit["af_per_anc"].values
    ).min(axis=1)
    dset_gwas_hit = dset_gwas_hit.sel(snp=maf > 0.01)

    # load hm3 data set
    dset_hm3 = common.load_page_hm3()

    # downsample
    df_gwas_hit = dset_gwas_hit.snp.to_dataframe().iloc[::10, :]

    imputed_vcf_dir = (
        "/u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s01_dataset/all"
    )
    row = df_gwas_hit.iloc[region_i, :]
    dset = admix.io.read_vcf(
        join(imputed_vcf_dir, f"chr{row.CHROM}.vcf.gz"),
        region=f"{row.CHROM}:{row.POS - 50_000}-{row.POS + 50_000}",
    ).sel(indiv=dset_hm3.indiv.values)
    dset = admix.data.impute_lanc(dset=dset, dset_ref=dset_hm3)
    dset.attrs["n_anc"] = 2
    admix.tools.allele_per_anc(dset)
    admix.tools.af_per_anc(dset)

    # only use SNPs with MAF > 0.001 in both ancestries
    dset = dset.sel(snp=np.all(dset.af_per_anc > 0.001, axis=1))

    dict_rls = {}

    for test_snp_rank in [0, 1, 3, 5]:
        rls_list = []
        allele_per_anc = dset["allele_per_anc"].values

        # 1. simulate a single variant and run GWAS
        sim = admix.simulate.continuous_pheno(
            dset=dset, var_g=0.1, var_e=1.0, n_causal=1, n_sim=50
        )
        beta, pheno_g, pheno = sim["beta"], sim["pheno_g"], sim["pheno"]

        # 2. select the index variant and run heterogeneity test
        n_sim = pheno.shape[1]
        for sim_i in tqdm(range(n_sim)):
            sim_beta = beta[:, :, sim_i]
            sim_pheno = pheno[:, sim_i]
            causal_snp = np.where(sim_beta[:, 0])[0].item()

            att_assoc = admix.assoc.marginal_fast(
                dset=dset.assign_coords(pheno=("indiv", sim_pheno)),
                pheno="pheno",
                method="ATT",
                family="linear",
            )

            tractor_assoc = admix.assoc.marginal_fast(
                dset=dset.assign_coords(pheno=("indiv", sim_pheno)),
                pheno="pheno",
                method="TRACTOR",
                family="linear",
            )
            # use att_assoc to determine the test_snp

            test_snp = np.argsort(att_assoc.P)[test_snp_rank]

            test_het_pval, test_het_model = common.test_het(
                allele_per_anc[:, test_snp, :], sim_pheno
            )
            causal_het_pval, causal_het_model = common.test_het(
                allele_per_anc[:, causal_snp, :], sim_pheno
            )
            rls_list.append(
                {
                    "causal_snp": causal_snp,
                    "test_snp": test_snp,
                    "att": att_assoc,
                    "tractor": tractor_assoc,
                    "test_het_pval": test_het_pval,
                    "causal_het_pval": causal_het_pval,
                }
            )

        dict_rls[test_snp_rank] = rls_list

    with open(join(out_dir, f"region_{region_i}.pkl"), "wb") as f:
        pickle.dump(dict_rls, f)


if __name__ == "__main__":
    fire.Fire(run)