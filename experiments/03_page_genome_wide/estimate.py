import xarray as xr
import numpy as np
import admix
import pandas as pd
from admix.data import quantile_normalize
from os.path import join
import pickle
import sys

sys.path.append("../../")
import common
import fire


def estimate_rg(est, est_var):
    """
    Estimate the ratio of genetic correlation.
    est: (3, ) numpy array
    est_var (3, 3) variance-covariance matrix
    """
    x, y = est[0], est[1]
    rg = y / x
    # grad = [-y / x^2, 1 / x]
    grad = np.array([-y / (x ** 2), 1 / x])

    def quad_form(x, A):
        return np.dot(np.dot(x.T, A), x)

    return rg, quad_form(grad, est_var[0:2, 0:2])


def estimate(trait_i, out_dir):

    K1 = np.load("out/admix_grm/K1.all.npy")
    K2 = np.load("out/admix_grm/K2.all.npy")
    K12 = np.load("out/admix_grm/K12.all.npy")

    dset = common.load_page_hm3()
    dset["A1"] = (("indiv", "indiv"), K1 + K2)
    dset["A2"] = (("indiv", "indiv"), K12 + K12.T)

    SUPP_TABLE_URL = "supp_tables.xlsx"
    trait_list = pd.read_excel(SUPP_TABLE_URL, sheet_name="trait-info")["trait"].values
    trait = trait_list[trait_i]

    dset_cor = dset.sel(indiv=~np.isnan(dset[f"{trait}"].values))
    study_dummies = pd.get_dummies(dset_cor["study"], drop_first=True)
    for c in study_dummies:
        dset_cor[f"study_dummy_{c}"] = ("indiv", study_dummies[c])
    study_dummy_cols = [f"study_dummy_{c}" for c in study_dummies]

    pheno = dset_cor[f"{trait}"].values
    pheno = quantile_normalize(pheno)
    est, est_var = admix.estimate.admix_gen_cor(
        dset=dset_cor,
        pheno=pheno,
        cov_cols=["age", "sex"]
        + study_dummy_cols
        + [f"geno_EV{i}" for i in range(1, 11)],
    )[0]
    print("--------------")
    print(f"{trait} (N={dset_cor.dims['indiv']})")
    rg, rg_var = estimate_rg(est, est_var)

    dict_rls = {
        "trait": trait,
        "n_indiv": dset_cor.dims["indiv"],
        "est": est,
        "est_var": est_var,
        "rg": rg,
        "rg_var": rg_var,
    }

    with open(join(out_dir, f"trait_{trait}.pkl"), "wb") as f:
        pickle.dump(dict_rls, f)


if __name__ == "__main__":
    fire.Fire(estimate)
