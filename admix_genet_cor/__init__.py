import xarray as xr
import numpy as np
from typing import List, Tuple
import dask.array as da
import dask
import pandas as pd
from scipy import linalg
from tqdm import tqdm
from .locus import test_snp_het, test_snp_assoc
from .utils import load_page_hm3

def calc_snp_prior_var(df_snp_info, her_model):
    """
    Calculate the SNP prior variance from SNP information
    """
    assert her_model in ["uniform", "gcta", "ldak", "mafukb"]
    if her_model == "uniform":
        return np.ones(len(df_snp_info))
    elif her_model == "gcta":
        freq = df_snp_info["FREQ"].values
        assert np.all(freq > 0), "frequencies should be larger than zero"
        return np.float_power(freq * (1 - freq), -1)
    elif her_model == "mafukb":
        # MAF-dependent genetic architecture, \alpha = -0.38 estimated from meta-analysis in UKB traits
        freq = df_snp_info["FREQ"].values
        assert np.all(freq > 0), "frequencies should be larger than zero"
        return np.float_power(freq * (1 - freq), -0.38)
    elif her_model == "ldak":
        freq, weight = df_snp_info["FREQ"].values, df_snp_info["LDAK_WEIGHT"].values
        return np.float_power(freq * (1 - freq), -0.25) * weight
    else:
        raise NotImplementedError
        
def af_per_anc(geno, lanc, n_anc=2) -> np.ndarray:
    """
    Calculate allele frequency per ancestry

    If at one particular SNP locus, no SNP from one particular ancestry can be found
    the corresponding entries will be filled with np.NaN.

    Parameters
    ----------
    dset: xr.Dataset
        Containing geno, lanc, n_anc

    Returns
    -------
    List[np.ndarray]
        `n_anc` length list of allele frequencies.
    """
    assert np.all(geno.shape == lanc.shape)
    n_snp = geno.shape[0]
    af = np.zeros((n_snp, n_anc))

    snp_chunks = geno.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in range(len(indices) - 1):
        start, stop = indices[i], indices[i + 1]
        geno_chunk = geno[start:stop, :, :].compute()
        lanc_chunk = lanc[start:stop, :, :].compute()

        for anc_i in range(n_anc):
            # mask SNPs with local ancestry not `i_anc`
            af[start:stop, anc_i] = (
                np.ma.masked_where(lanc_chunk != anc_i, geno_chunk)
                .sum(axis=(1, 2))
                .data
            ) / np.sum(lanc_chunk == anc_i, axis=(1, 2))

    return af


def allele_per_anc(geno, lanc, center=False, n_anc=2):
    """Get allele count per ancestry

    Parameters
    ----------
    ds: xr.Dataset
        Containing geno, lanc, n_anc
    center: bool
        whether to center the data around empirical frequencies of each ancestry
    inplace: bool
        whether to return a new dataset or modify the input dataset
    Returns
    -------
    Return allele counts per ancestries
    """
    assert np.all(geno.shape == lanc.shape), "shape of `hap` and `lanc` are not equal"
    assert geno.ndim == 3, "`hap` and `lanc` should have three dimension"
    n_snp, n_indiv, n_haplo = geno.shape
    assert n_haplo == 2, "`n_haplo` should equal to 2, check your data"

    assert isinstance(geno, da.Array) & isinstance(
        lanc, da.Array
    ), "`geno` and `lanc` should be dask array"

    # make sure the chunk size along the ploidy axis to be 2
    geno = geno.rechunk({2: 2})
    lanc = lanc.rechunk({2: 2})

    # rechunk so that all chunk of `n_anc` is passed into the helper function
    assert (
        n_anc == 2
    ), "`n_anc` should be 2, NOTE: not so clear what happens when `n_anc = 3`"

    assert (
        geno.chunks == lanc.chunks
    ), "`geno` and `lanc` should have the same chunk size"

    assert (
        len(geno.chunks[1]) == 1
    ), "geno / lanc should not be chunked across individual dimension"

    def helper(geno_chunk, lanc_chunk, n_anc, center):

        n_snp, n_indiv, n_haplo = geno_chunk.shape
        if center:
            af_chunk = af_per_anc(
                da.from_array(geno_chunk), da.from_array(lanc_chunk), n_anc=n_anc
            )
        else:
            af_chunk = None
        apa = np.zeros((n_snp, n_indiv, n_anc), dtype=np.float64)
        for i_haplo in range(n_haplo):
            haplo_hap = geno_chunk[:, :, i_haplo]
            haplo_lanc = lanc_chunk[:, :, i_haplo]
            for i_anc in range(n_anc):
                if af_chunk is None:
                    apa[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[
                        haplo_lanc == i_anc
                    ]
                else:
                    # for each SNP, find the corresponding allele frequency
                    apa[:, :, i_anc][haplo_lanc == i_anc] += (
                        haplo_hap[haplo_lanc == i_anc]
                        - af_chunk[np.where(haplo_lanc == i_anc)[0], i_anc]
                    )
        return apa

    rls_allele_per_anc = da.map_blocks(
        lambda geno_chunk, lanc_chunk: helper(
            geno_chunk=geno_chunk, lanc_chunk=lanc_chunk, n_anc=n_anc, center=center
        ),
        geno,
        lanc,
        dtype=np.float64,
    )

    return rls_allele_per_anc


def simulate_continuous_pheno(
    geno,
    lanc,
    n_anc=2,
    hsq: float = None,
    cor: float = None,
    n_causal: int = None,
    beta: np.ndarray = None,
    snp_prior_var: np.ndarray = None,
    n_sim=10,
    apa_center=False,
) -> dict:
    """Simulate continuous phenotype of admixed individuals [continuous]

    Parameters
    ----------
    dset: xr.Dataset
        Dataset containing the following variables:
            - geno: (n_indiv, n_snp, 2) phased genotype of each individual
            - lanc: (n_indiv, n_snp, 2) local ancestry of each SNP
    hsq: float
        Variance explained by genotypical effects
    var_e: float
        Variance explained by the effect of the environment
    gamma: float
        Correlation between the genetic effects from two ancestral backgrounds
    n_causal: int, optional
        number of causal variables, by default None
    beta: np.ndarray, optional
        causal effect of each causal variable, by default None
    cov_cols: List[str], optional
        list of covariates to include as covariates, by default None
    cov_effects: List[float], optional
        list of the effect of each covariate, by default None
        for each simulation, the cov_effects will be the same
    n_sim : int, optional
        number of simulations, by default 10


    Returns
    -------
    beta: np.ndarray
        simulated effect sizes (2 * n_snp, n_sim)
    phe_g: np.ndarray
        simulated genetic component of phenotypes (n_indiv, n_sim)
    phe: np.ndarray
        simulated phenotype (n_indiv, n_sim)
    """
    assert n_anc == 2, "Only two-ancestry currently supported"
    apa = allele_per_anc(geno, lanc, center=apa_center)
    n_snp, n_indiv = apa.shape[0:2]

    # simulate effect sizes
    if beta is None:
        if cor is None:
            cor = 1.0

        if n_causal is None:
            # n_causal = n_snp if `n_causal` is not specified
            n_causal = n_snp

        assert (-1 <= cor) and (
            cor <= 1
        ), "Correlation parameter should be between [-1, 1]"

        assert n_causal <= n_snp, "n_causal must be <= n_snp"

        if snp_prior_var is None:
            snp_prior_var = np.ones(n_snp)

        # if `beta` is not specified, simulate effect sizes
        beta = np.zeros((n_snp, n_anc, n_sim))
        for i_sim in range(n_sim):
            cau = sorted(
                np.random.choice(np.arange(n_snp), size=n_causal, replace=False)
            )

            i_beta = np.random.multivariate_normal(
                mean=[0.0, 0.0],
                cov=np.array([[1, cor], [cor, 1]]),
                size=n_causal,
            )

            i_beta = i_beta * np.sqrt(snp_prior_var[cau])[:, None]

            for i_anc in range(n_anc):
                beta[cau, i_anc, i_sim] = i_beta[:, i_anc]
    else:
        assert (cor is None) and (
            n_causal is None
        ), "If `beta` is specified, `cor`, and `n_causal` must not be specified"
        assert beta.shape == (n_snp, n_anc) or beta.shape == (
            n_snp,
            n_anc,
            n_sim,
        ), "`beta` must be of shape (n_snp, n_anc) or (n_snp, n_anc, n_sim)"
        if beta.shape == (n_snp, n_anc):
            # replicate `beta` for each simulation
            beta = np.repeat(beta[:, :, np.newaxis], n_sim, axis=2)

    pheno_g = np.zeros([n_indiv, n_sim])
    snp_chunks = apa.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(
        range(len(indices) - 1), desc="admix_genet_cor.simulate_continuous_pheno"
    ):
        start, stop = indices[i], indices[i + 1]
        apa_chunk = apa[start:stop, :, :].compute()
        for i_anc in range(n_anc):
            pheno_g += np.dot(apa_chunk[:, :, i_anc].T, beta[start:stop, i_anc, :])

    # scale variance of pheno_g to hsq
    std_scale = np.sqrt(hsq / np.var(pheno_g, axis=0))
    pheno_g *= std_scale
    beta *= std_scale

    pheno_e = np.zeros(pheno_g.shape)
    for i_sim in range(n_sim):
        pheno_e[:, i_sim] = np.random.normal(
            loc=0.0, scale=np.sqrt(1 - hsq), size=n_indiv
        )

    pheno = pheno_g + pheno_e
    return {"beta": beta, "pheno_g": pheno_g, "pheno": pheno}


def compute_grm(geno, lanc, n_anc=2, snp_prior_var=None, apa_center=False):
    """Calculate ancestry specific GRM matrix

    Parameters
    ----------
    center: bool
        whether to center the `allele_per_ancestry` matrix
        in the calculation
    inplace: bool
        whether to return a new dataset or modify the input dataset

    Returns
    -------
    If `inplace` is False, return a dictionary of GRM matrices
        - K1: np.ndarray
            ancestry specific GRM matrix for the 1st ancestry
        - K2: np.ndarray
            ancestry specific GRM matrix for the 2nd ancestry
        - K12: np.ndarray
            ancestry specific GRM matrix for cross term of the 1st and 2nd ancestry
    """

    assert n_anc == 2, "only two-way admixture is implemented"
    assert np.all(geno.shape == lanc.shape)

    apa = allele_per_anc(geno, lanc, center=apa_center)
    n_snp, n_indiv = apa.shape[0:2]

    if snp_prior_var is None:
        snp_prior_var = np.ones(n_snp)

    K1 = np.zeros([n_indiv, n_indiv])
    K2 = np.zeros([n_indiv, n_indiv])
    K12 = np.zeros([n_indiv, n_indiv])

    snp_chunks = apa.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(range(len(indices) - 1), desc="admix_genet_cor.compute_grm"):
        start, stop = indices[i], indices[i + 1]
        apa_chunk = apa[start:stop, :, :].compute()

        # multiply by the prior variance on each SNP
        apa_chunk *= np.sqrt(snp_prior_var[start:stop])[:, None, None]
        a1_chunk, a2_chunk = apa_chunk[:, :, 0], apa_chunk[:, :, 1]

        K1 += np.dot(a1_chunk.T, a1_chunk) / sum(snp_prior_var)
        K2 += np.dot(a2_chunk.T, a2_chunk) / sum(snp_prior_var)
        K12 += np.dot(a1_chunk.T, a2_chunk) / sum(snp_prior_var)

    return K1, K2, K12


def estimate_genetic_cor(
    A1, A2, pheno: np.ndarray, cov: np.ndarray = None, compute_varcov: bool = False
):
    """Estimate genetic correlation given a dataset, phenotypes, and covariates.
    This is a very specialized function that tailed for estimating the genetic correlation
    for variants in different local ancestry backgrounds.

    See details in https://www.nature.com/articles/s41467-020-17576-9#MOESM1

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to estimate correlation from.
    pheno: np.ndarray
        Phenotypes to estimate genetic correlation. If a matrix is provided, then each
        column is treated as a separate phenotype.
    cov_cols: list, optional
        List of covariate columns.
    cov_intercept: bool, optional
        Whether to include intercept in covariate matrix.
    """
    assert np.all(A1.shape == A2.shape), "`A1` and `A2` must have the same shape"
    n_indiv = A1.shape[0]

    # build projection matrix from covariate matrix
    if cov is None:
        cov_proj_mat = np.eye(n_indiv)
    else:
        cov_proj_mat = np.eye(n_indiv) - np.linalg.multi_dot(
            [cov, np.linalg.inv(np.dot(cov.T, cov)), cov.T]
        )

    if pheno.ndim == 1:
        pheno = pheno.reshape((-1, 1))
    assert pheno.shape[0] == n_indiv

    n_pheno = pheno.shape[1]

    pheno = np.dot(cov_proj_mat, pheno)
    quad_form_func = lambda x, A: np.dot(np.dot(x.T, A), x)

    grm_list = [A1, A2, np.eye(n_indiv)]
    grm_list = [cov_proj_mat @ grm @ cov_proj_mat for grm in grm_list]

    # multiply cov_proj_mat
    n_grm = len(grm_list)
    design = np.zeros((n_grm, n_grm))
    for i in range(n_grm):
        for j in range(n_grm):
            if i <= j:
                design[i, j] = (grm_list[i] * grm_list[j]).sum()
                design[j, i] = design[i, j]

    rls_list: List[Tuple] = []
    for i_pheno in tqdm(range(n_pheno)):
        response = np.zeros(n_grm)
        for i in range(n_grm):
            response[i] = quad_form_func(pheno[:, i_pheno], grm_list[i])

        # point estimate
        var_comp = linalg.solve(
            design,
            response,
        )

        if compute_varcov:
            # variance-covariance matrix
            inv_design = linalg.inv(design)
            Sigma = np.zeros_like(grm_list[0])
            for i in range(n_grm):
                Sigma += var_comp[i] * grm_list[i]
            Sigma_grm_list = [np.dot(Sigma, grm) for grm in grm_list]

            var_response = np.zeros((n_grm, n_grm))
            for i in range(n_grm):
                for j in range(n_grm):
                    if i <= j:
                        var_response[i, j] = (
                            2 * (Sigma_grm_list[i] * Sigma_grm_list[j]).sum()
                        )
                        var_response[j, i] = var_response[i, j]
            var_comp_var = np.linalg.multi_dot([inv_design, var_response, inv_design])
            rls_list.append((var_comp, var_comp_var))
        else:
            rls_list.append(var_comp)
    return rls_list