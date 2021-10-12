import xarray as xr
import numpy as np
from typing import List, Tuple
import admix
import dask.array as da
import dask
from typing import Optional
import pandas as pd
from scipy import linalg
from tqdm import tqdm


def af_per_anc(dset: xr.Dataset, inplace=True) -> Optional[np.ndarray]:
    """
    Calculate allele frequency per ancestry

    Parameters
    ----------
    dset: xr.Dataset
        Containing geno, lanc, n_anc

    Returns
    -------
    List[np.ndarray]
        `n_anc` length list of allele frequencies.
    """
    assert "geno" in dset.data_vars, "`geno` not in `ds.data_vars`"
    assert "lanc" in dset.data_vars, "`lanc` not in `ds.data_vars`"
    n_anc = dset.attrs["n_anc"]
    geno = dset.data_vars["geno"]
    lanc = dset.data_vars["lanc"]
    rls = []

    for i_anc in range(n_anc):
        # mask SNPs with local ancestry not `i_anc`
        rls.append(
            da.ma.getdata(
                da.ma.masked_where(lanc != i_anc, geno).mean(axis=(1, 2))
            ).compute()
        )
    rls = da.from_array(np.array(rls)).T
    if inplace:
        dset["af_per_anc"] = xr.DataArray(
            rls,
            dims=(
                "snp",
                "anc",
            ),
        )
        return None
    else:
        return rls


def allele_per_anc(ds, center=False, inplace=True):
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
    geno, lanc = ds.data_vars["geno"].data, ds.data_vars["lanc"].data

    n_anc = ds.attrs["n_anc"]
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

    # TODO: align the chunk size along 1st axis to be the same

    def helper(geno_chunk, lanc_chunk, n_anc, af_chunk=None):

        n_snp, n_indiv, n_haplo = geno_chunk.shape
        assert af_chunk.shape[0] == n_snp
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
                    apa[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[
                        haplo_lanc == i_anc
                    ] - af_chunk[np.where(haplo_lanc == i_anc)[0], :, i_anc].squeeze(
                        axis=1
                    )
        return apa

    if center:
        if inplace:
            af_per_anc(ds, inplace=True)
            af = ds.data_vars["af_per_anc"].data
        else:
            af = af_per_anc(ds, inplace=False)
        # rechunk so that all chunk of `n_anc` is passed into the helper function
        assert (
            n_anc == 2
        ), "`n_anc` should be 2, NOTE: not so clear what happens when `n_anc = 3`"

        assert (
            geno.chunks == lanc.chunks
        ), "`geno` and `lanc` should have the same chunk size"

        af = af.rechunk({0: geno.chunks[0], 1: n_anc})

        rls_allele_per_anc = da.map_blocks(
            lambda geno_chunk, lanc_chunk, af_chunk: helper(
                geno_chunk=geno_chunk,
                lanc_chunk=lanc_chunk,
                n_anc=n_anc,
                af_chunk=af_chunk,
            ),
            geno,
            lanc,
            af[:, None, :],
            dtype=np.float64,
        )

    else:
        rls_allele_per_anc = da.map_blocks(
            lambda geno_chunk, lanc_chunk: helper(
                geno_chunk=geno_chunk, lanc_chunk=lanc_chunk, n_anc=n_anc
            ),
            geno,
            lanc,
            dtype=np.float64,
        )
    if inplace:
        ds["allele_per_anc"] = xr.DataArray(
            rls_allele_per_anc, dims=("snp", "indiv", "anc")
        )
    else:
        return rls_allele_per_anc


def simulate_continuous_pheno(
    dset: xr.Dataset,
    var_g: float = None,
    var_e: float = None,
    gamma: float = None,
    n_causal: int = None,
    beta: np.ndarray = None,
    cov_cols: List[str] = None,
    cov_effects: List[float] = None,
    n_sim=10,
) -> dict:
    """Simulate continuous phenotype of admixed individuals [continuous]

    Parameters
    ----------
    dset: xr.Dataset
        Dataset containing the following variables:
            - geno: (n_indiv, n_snp, 2) phased genotype of each individual
            - lanc: (n_indiv, n_snp, 2) local ancestry of each SNP
    var_g: float or np.ndarray
        Variance explained by the genotype effect
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
    n_anc = dset.n_anc
    assert n_anc == 2, "Only two-ancestry currently supported"

    # TODO: center or not is really critical here, and should be carefully thought
    if "allele_per_anc" not in dset.data_vars:
        allele_per_anc(dset, center=True)

    apa = dset.data_vars["allele_per_anc"]
    n_snp, n_indiv = apa.shape[0:2]

    # simulate effect sizes
    if beta is None:
        if gamma is None:
            # covariance of effects across ancestries set to 1 if `gamma` is not specfied.
            gamma = var_g

        if n_causal is None:
            # n_causal = n_snp if `n_causal` is not specified
            n_causal = n_snp

        # if `beta` is not specified, simulate effect sizes
        beta = np.zeros((n_snp, n_anc, n_sim))
        for i_sim in range(n_sim):
            cau = sorted(
                np.random.choice(np.arange(n_snp), size=n_causal, replace=False)
            )

            expected_cov = np.array([[var_g, gamma], [gamma, var_g]]) / n_causal

            i_beta = np.random.multivariate_normal(
                mean=[0.0, 0.0],
                cov=expected_cov,
                size=n_causal,
            )
            # normalize to expected covariance structure
            empirical_cov = np.dot(i_beta.T, i_beta) / n_causal
            i_beta = i_beta * np.sqrt(np.diag(expected_cov) / np.diag(empirical_cov))

            for i_anc in range(n_anc):
                beta[cau, i_anc, i_sim] = i_beta[:, i_anc]
    else:
        assert (
            (var_g is None) and (gamma is None) and (n_causal is None)
        ), "If `beta` is specified, `var_g`, `var_e`, `gamma`, and `n_causal` must be specified"
        assert beta.shape == (n_snp, n_anc) or beta.shape == (
            n_snp,
            n_anc,
            n_sim,
        ), "`beta` must be of shape (n_snp, n_anc) or (n_snp, n_anc, n_sim)"
        if beta.shape == (n_snp, n_anc):
            # replicate `beta` for each simulation
            beta = np.repeat(beta[:, :, np.newaxis], n_sim, axis=2)

    pheno_g = da.zeros([n_indiv, n_sim])
    for i_anc in range(n_anc):
        pheno_g += da.dot(apa[:, :, i_anc].T, beta[:, i_anc, :])

    pheno_e = np.zeros(pheno_g.shape)
    for i_sim in range(n_sim):
        pheno_e[:, i_sim] = np.random.normal(
            loc=0.0, scale=np.sqrt(var_e), size=n_indiv
        )

    pheno = pheno_g + pheno_e
    pheno_g, pheno = dask.compute((pheno_g, pheno))[0]
    # if `cov_cols` are specified, add the covariates to the phenotype
    if cov_cols is not None:
        # if `cov_effects` are not set, set to random normal values
        if cov_effects is None:
            cov_effects = np.random.normal(size=len(cov_cols))
        # add the covariates to the phenotype
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset[cov_col].values
        pheno += np.dot(cov_values, cov_effects).reshape((n_indiv, 1))

    return {
        "beta": beta,
        "pheno_g": pheno_g,
        "pheno": pd.DataFrame(
            pheno, index=dset.indiv.values, columns=[f"SIM_{i}" for i in range(n_sim)]
        ),
        "cov_effects": cov_effects,
    }


def compute_grm(
    dset,
    center: bool = False,
):
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

    geno = dset["geno"].data
    lanc = dset["lanc"].data
    n_anc = dset.attrs["n_anc"]
    assert n_anc == 2, "only two-way admixture is implemented"
    assert np.all(geno.shape == lanc.shape)

    apa = allele_per_anc(dset, center=center, inplace=False).astype(float)

    n_snp, n_indiv = apa.shape[0:2]

    K1 = np.zeros([n_indiv, n_indiv])
    K2 = np.zeros([n_indiv, n_indiv])
    K12 = np.zeros([n_indiv, n_indiv])

    snp_chunks = apa.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(range(len(indices) - 1), desc="admix_genet_cor.compute_grm"):
        start, stop = indices[i], indices[i + 1]
        apa_chunk = apa[start:stop, :, :].compute()
        a1_chunk, a2_chunk = apa_chunk[:, :, 0], apa_chunk[:, :, 1]
        K1 += np.dot(a1_chunk.T, a1_chunk) / n_snp
        K2 += np.dot(a2_chunk.T, a2_chunk) / n_snp
        K12 += np.dot(a1_chunk.T, a2_chunk) / n_snp

    return K1, K2, K12


def estimate_genetic_cor(
    dset: xr.Dataset,
    pheno: np.ndarray,
    cov_cols: List[str] = None,
    cov_intercept: bool = True,
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
    n_indiv = dset.dims["indiv"]

    # build the covariate matrix
    if cov_cols is not None:
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset[cov_col].values
        if cov_intercept:
            cov_values = np.c_[np.ones((n_indiv, 1)), cov_values]
    else:
        cov_values = None
        if cov_intercept:
            cov_values = np.ones((n_indiv, 1))

    # build projection matrix from covariate matrix
    if cov_values is None:
        cov_proj_mat = np.eye(n_indiv)
    else:
        cov_proj_mat = np.eye(n_indiv) - np.linalg.multi_dot(
            [cov_values, np.linalg.inv(np.dot(cov_values.T, cov_values)), cov_values.T]
        )

    if pheno.ndim == 1:
        pheno = pheno.reshape((-1, 1))
    assert pheno.shape[0] == n_indiv

    n_pheno = pheno.shape[1]

    pheno = np.dot(cov_proj_mat, pheno)
    quad_form_func = lambda x, A: np.dot(np.dot(x.T, A), x)
    n_indiv = dset.dims["indiv"]
    n_snp = dset.dims["snp"]

    grm_list = [dset["A1"].data, dset["A2"].data, np.eye(n_indiv)]
    grm_list = [np.dot(grm, cov_proj_mat) for grm in grm_list]

    # multiply cov_proj_mat
    n_grm = len(grm_list)
    design = np.zeros((n_grm, n_grm))
    for i in range(n_grm):
        for j in range(n_grm):
            if i <= j:
                design[i, j] = (grm_list[i] * grm_list[j]).sum()
                design[j, i] = design[i, j]

    rls_list: List[Tuple] = []
    for i_pheno in range(n_pheno):
        response = np.zeros(n_grm)
        for i in range(n_grm):
            response[i] = quad_form_func(pheno[:, i_pheno], grm_list[i])

        # point estimate
        var_comp = linalg.solve(
            design,
            response,
        )

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
    return rls_list