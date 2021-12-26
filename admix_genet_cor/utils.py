import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import sparse
import numpy as np
import fire
from os.path import join
from typing import List
import fire
import statsmodels.api as sm
import dask.array as da
from admix.data import af_per_anc


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
    assert center is False, "`center` is not implemented"
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


def simulate_quant_pheno(
    geno: da.Array,
    lanc: da.Array,
    hsq: float,
    n_anc: int = 2,
    cor: float = None,
    n_causal: int = None,
    beta: np.ndarray = None,
    snp_prior_var: np.ndarray = None,
    n_sim=10,
) -> dict:
    """Simulate continuous phenotype of admixed individuals [continuous]

    Parameters
    ----------
    geno: da.Array
        (n_indiv, n_snp, 2) phased genotype of each individual
    lanc: da.Array
        (n_indiv, n_snp, 2) local ancestry of each SNP
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
    apa = allele_per_anc(geno, lanc, n_anc=n_anc)
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

        # TODO: also change to sparse implementation
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
        assert beta.shape == (
            n_snp,
            n_anc,
            n_sim,
        ), "`beta` must be of shape (n_snp, n_anc, n_sim)"
        # if beta.shape == (n_snp, n_anc):
        #     # replicate `beta` for each simulation
        #     beta = np.repeat(beta[:, :, np.newaxis], n_sim, axis=2)

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


# loci heterogeneity analysis
def simulate_het(apa, beta, cov):
    cov_effects = np.random.normal(loc=0, scale=0.1, size=cov.shape[1])
    y = (
        np.dot(apa, beta)
        + np.dot(cov, cov_effects)
        + np.random.normal(size=apa.shape[0])
    )
    return y
