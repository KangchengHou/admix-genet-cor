from admix.data import read_int_mat, write_int_mat
import pandas as pd
from os.path import join
import numpy as np
from scipy import linalg
from os.path import exists, join
from utils import *
import fire
import zarr


def allele_count_per_anc(hap: np.ndarray, lanc: np.ndarray, n_anc: int):
    """Get allele count per ancestry

    Parameters
    ----------
    hap : np.ndarray
        haplotype (n_indiv, n_snp, n_anc)
    lanc : np.ndarray
        local ancestry (n_indiv, n_snp, n_anc)
    """
    assert np.all(hap.shape == lanc.shape), "shape of `hap` and `lanc` are not equal"
    assert hap.ndim == 3, "`hap` and `lanc` should have three dimension"
    n_indiv, n_snp, n_haplo = hap.shape
    assert n_haplo == 2, "`n_haplo` should equal to 2, check your data"
    geno = np.zeros((n_indiv, n_snp, n_anc), dtype=np.int8)

    for i_haplo in range(n_haplo):
        haplo_hap = hap[:, :, i_haplo]
        haplo_lanc = lanc[:, :, i_haplo]
        for i_anc in range(n_anc):
            geno[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[haplo_lanc == i_anc]

    return geno


def simulate_phenotype(
    hap: np.ndarray,
    lanc: np.ndarray,
    var_g: float,
    var_e: float,
    gamma: float,
    n_causal: int = None,
    ganc: np.ndarray = None,
    ganc_effect: float = None,
    n_sim=500,
    n_anc=2,
):
    """Simulate phenotype for admixture population [continuous]
    Parameters
    ----------
    hap : np.ndarray
        phased genotype (n_indiv, n_snp, 2)
    lanc : np.ndarray
        local ancestry (n_indiv, n_snp, 2), same as `hap`
    h2g : float
        desired heritability
    n_causal : int, optional
        number of causal variables, by default None
    ganc : np.ndarray, optional
        vector of global ancestry, by default None
    ganc_effect : float, optional
        global ancestry effect, by default None
    cov : float, optional
        covariance of genetic effect, by default 0.0
    n_sim : int, optional
        number of simulations, by default 30
    Returns
    -------
    beta : np.ndarray
        simulated effect sizes (2 * n_snp, n_sim)
    phe_g: np.ndarray
        simulated genetic component of phenotypes (n_indiv, n_sim)
    phe: np.ndarray
        simulated phenotype (n_indiv, n_sim)
    """
    geno = allele_count_per_anc(hap, lanc, n_anc=n_anc)
    n_indiv, n_snp = geno.shape[0], geno.shape[1]

    if ganc is not None:
        assert len(ganc) == n_indiv
    betas = np.zeros((n_snp, n_anc, n_sim))
    for i_sim in range(n_sim):
        cau = sorted(np.random.choice(np.arange(n_snp), size=n_causal, replace=False))

        # TODO: cope with multiple ancestry situation, here we only cope with 2.
        beta = np.random.multivariate_normal(
            mean=[0.0, 0.0],
            cov=np.array([[var_g, gamma], [gamma, var_g]]) / n_snp,
            size=n_causal,
        )
        for i_anc in range(n_anc):
            betas[cau, i_anc, i_sim] = beta[:, i_anc]

    phe_g = np.zeros([n_indiv, n_sim])
    for i_anc in range(n_anc):
        phe_g += np.dot(geno[:, :, i_anc], betas[:, i_anc, :])
    phe_e = np.zeros_like(phe_g)

    for sim_i in range(n_sim):
        phe_e[:, sim_i] = np.random.normal(loc=0.0, scale=np.sqrt(var_e), size=n_indiv)

    phe = phe_g + phe_e
    if ganc is not None:
        phe += np.dot(ganc[:, np.newaxis], ganc_effect * np.ones((1, n_sim)))
    return betas, phe_g, phe


def main(geno, var_g, var_e, gamma, out_prefix, p_causal=1.0, n_sim=100, seed=1234):
    np.random.seed(seed)
    dataset = zarr.load(geno)
    admix_hap = dataset["hap"]
    admix_lanc = dataset["lanc"]
    assert np.all(admix_hap.shape == admix_lanc.shape)
    n_indiv = admix_hap.shape[0]
    n_snp = admix_hap.shape[1]

    beta, phe_g, phe = simulate_phenotype(
        hap=admix_hap,
        lanc=admix_lanc,
        var_g=var_g,
        var_e=var_e,
        gamma=gamma,
        n_causal=int(p_causal * n_snp),
        n_sim=n_sim,
    )

    # write
    np.save(out_prefix + ".beta", beta)
    df_phen = {"sample_0": np.arange(n_indiv), "sample_1": np.arange(n_indiv)}
    for i_sim in range(phe.shape[1]):
        df_phen[f"phen_{i_sim}"] = phe[:, i_sim]
    df_phen = pd.DataFrame(df_phen)
    df_phen.to_csv(
        out_prefix + ".phen",
        sep="\t",
        index=False,
        header=False
        # , float_format="%.6f"
    )


if __name__ == "__main__":
    fire.Fire(main)
