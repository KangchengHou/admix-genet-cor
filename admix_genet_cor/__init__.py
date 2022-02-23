import xarray as xr
import numpy as np
from typing import List, Tuple
import dask.array as da
import dask
import pandas as pd
from scipy import linalg
from tqdm import tqdm
from .locus import (
    test_snp_het,
    test_snp_assoc,
    marginal_het,
    calc_cov,
    calc_apa_cov,
    simulate_hetero_assoc,
)
from .utils import simulate_quant_pheno, af_per_anc, allele_per_anc, hdi
from admix.data import calc_snp_prior_var


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