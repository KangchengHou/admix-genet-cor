import numpy as np
import allel
import xarray as xr
import pandas as pd
from tqdm import tqdm
import allel
import numpy as np
import fire
from os.path import join
from typing import List
import fire
import statsmodels.api as sm


# PAGE analysis
def load_page_hm3(
    chrom=None,
    GENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s03_aframr/dataset/hm3.zarr/",
    PHENO_DIR="/u/project/sgss/PAGE/phenotype/",
):
    if chrom is None:
        chrom = np.arange(1, 23)
    elif isinstance(chrom, List):
        chrom = np.array(chrom)
    elif isinstance(chrom, int):
        chrom = np.aarray([chrom])
    else:
        raise ValueError("chrom must be None, List or int")

    trait_cols = [
        # Inflammtory traits
        "crp",
        "total_wbc_cnt",
        "mean_corp_hgb_conc",
        "platelet_cnt",
        # lipid traits
        "hdl",
        "ldl",
        "triglycerides",
        "total_cholesterol",
        # lifestyle traits
        "cigs_per_day_excl_nonsmk_updated",
        "coffee_cup_day",
        # glycemic traits
        "a1c",
        "insulin",
        "glucose",
        "t2d_status",
        # electrocardiogram traits
        "qt_interval",
        "qrs_interval",
        "pr_interval",
        # blood pressure traits
        "systolic_bp",
        "diastolic_bp",
        "hypertension",
        # anthropometric traits
        "waist_hip_ratio",
        "height",
        "bmi",
        # kidney traits
        "egfrckdepi",
    ]

    covar_cols = ["study", "age", "sex", "race_ethnicity", "center"] + [
        f"geno_EV{i}" for i in range(1, 51)
    ]

    race_encoding = {
        1: "Unclassified",
        2: "African American",
        3: "Hispanic/Latino",
        4: "Asian",
        5: "Native Hawaiian",
        6: "Native American",
        7: "Other",
    }

    race_color = {
        "Unclassified": "#ffffff",
        "African American": "#e9d16a",
        "Hispanic/Latino": "#9a3525",
        "Asian": "#3c859d",
        "Native Hawaiian": "#959f6e",
        "Native American": "#546f7b",
        "Other": "#d07641",
    }

    df_pheno = pd.read_csv(
        join(PHENO_DIR, "MEGA_page-harmonized-phenotypes-pca-freeze2-2016-12-14.txt"),
        sep="\t",
        na_values=".",
        low_memory=False,
    )
    df_pheno["race_ethnicity"] = df_pheno["race_ethnicity"].map(race_encoding)

    dset_list = []
    for i_chr in tqdm(chrom):
        dset_list.append(xr.open_zarr(join(GENO_DIR, f"chr{i_chr}.zip")))

    dset = xr.concat(dset_list, dim="snp")

    df_aframr_pheno = df_pheno.set_index("PAGE_Subject_ID").loc[
        dset.indiv.values, trait_cols + covar_cols
    ]

    for col in df_aframr_pheno.columns:
        dset[f"{col}@indiv"] = ("indiv", df_aframr_pheno[col].values)

    for col in ["center", "study", "race_ethnicity"]:
        dset[f"{col}@indiv"] = dset[f"{col}@indiv"].astype(str)

    # format the dataset to follow the new standards
    for k in dset.data_vars.keys():
        if k.endswith("@indiv"):
            dset.coords[k.split("@")[0]] = ("indiv", dset.data_vars[k].data)
        if k.endswith("@snp"):
            dset.coords[k.split("@")[0]] = ("snp", dset.data_vars[k].data)
    dset = dset.drop_vars(
        [
            k
            for k in dset.data_vars.keys()
            if (k.endswith("@indiv") or k.endswith("@snp"))
        ]
    )

    dset = dset.rename({n: n.split("@")[0] for n in [k for k in dset.coords.keys()]})

    return dset


def concat_grm(out_dir="out/admix_grm"):
    dset = load_hm3()
    n_indiv = dset.dims["indiv"]
    n_total_snp = dset.dims["snp"]
    K1 = np.zeros((n_indiv, n_indiv))
    K2 = np.zeros((n_indiv, n_indiv))
    K12 = np.zeros((n_indiv, n_indiv))

    for chr_i in tqdm(range(1, 23)):
        n_chr_snp = np.char.startswith(dset["snp"].values, f"chr{chr_i}:").sum()
        K1 += np.load(join(out_dir, f"K1.chr{chr_i}.npy")) * n_chr_snp
        K2 += np.load(join(out_dir, f"K2.chr{chr_i}.npy")) * n_chr_snp
        K12 += np.load(join(out_dir, f"K12.chr{chr_i}.npy")) * n_chr_snp

    np.save(join(out_dir, "K1.all.npy"), K1 / n_total_snp)
    np.save(join(out_dir, "K2.all.npy"), K2 / n_total_snp)
    np.save(join(out_dir, "K12.all.npy"), K12 / n_total_snp)


# loci heterogeneity analysis


def simulate_het(apa, beta, cov):
    cov_effects = np.random.normal(loc=0, scale=0.1, size=cov.shape[1])
    y = (
        np.dot(apa, beta)
        + np.dot(cov, cov_effects)
        + np.random.normal(size=apa.shape[0])
    )
    return y




if __name__ == "__main__":
    fire.Fire()