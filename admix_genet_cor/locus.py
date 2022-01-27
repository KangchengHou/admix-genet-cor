"""
Functions for examining locus-level heterogeneity
"""
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import admix
import pandas as pd
import dapgen
import scipy
from .utils import simulate_quant_pheno


def calc_cov(m1, m2):
    # m1: (n_snp1, n_indiv)
    # m2: (n_snp2, n_indiv)
    assert m1.shape[1] == m2.shape[1]
    m1 = m1 - m1.mean(axis=1)[:, None]
    m2 = m2 - m2.mean(axis=1)[:, None]
    n_indiv = m1.shape[1]
    return np.dot(m1, m2.T) / n_indiv


def calc_apa_cov(apa, m):
    # apa: (n_snp, n_indiv, 2)
    # m: (n_indiv, )

    n_snp, n_indiv = apa.shape[0:2]
    assert apa.shape[1] == m.shape[0]
    # center apa
    apa = apa - apa.mean(axis=1)[:, None, :]
    m = m - m.mean()
    ex_list = np.zeros((n_snp, 2))
    z_list = np.zeros(n_snp)
    # for each snp, calculate the covariance
    transform_vec = np.array([[1], [-1]])
    for snp_i in tqdm(range(n_snp)):
        cov_inv = np.linalg.inv(np.cov(apa[snp_i, :, :], rowvar=False, ddof=0))
        # expectation and variance
        ex = cov_inv @ apa[snp_i, :, :].T @ m
        ex_list[snp_i, :] = ex
        z_list[snp_i] = (
            transform_vec.T @ ex / np.sqrt(transform_vec.T @ cov_inv @ transform_vec)
        )
    return z_list, ex_list


def test_snp_het(apa, y, cov=None):
    """
    Test heterogeneity level at individual SNPs
    apa: (n_indiv, 2)
    y: (n_indiv, )
    cov: (n_indiv, n_cov)
    """
    assert apa.shape[1] == 2
    if cov is not None:
        design = sm.add_constant(np.hstack([apa, cov]))
    else:
        design = sm.add_constant(apa)

    model = sm.OLS(y, design).fit()

    A = np.zeros([1, len(model.params)])
    A[0, 1] = 1
    A[0, 2] = -1
    p_ftest = model.f_test(A).pvalue.item()
    return p_ftest, model


def marginal_het(geno, lanc, y, cov=None):
    """
    Test heterogeneity level at individual SNPs
    apa: (n_indiv, 2)
    y: (n_indiv, )
    cov: (n_indiv, n_cov)
    """
    apa = admix.data.allele_per_anc(geno, lanc, n_anc=2).compute()
    n_snp, n_indiv, n_anc = apa.shape
    assert n_anc == 2, "only two-ancestry difference test is supported for now"
    if cov is None:
        design = np.ones((n_indiv, 1))
    else:
        design = np.hstack([np.ones((n_indiv, 1)), cov])
    # design: [intercept] [covariates] [anc1-allele, anc2-allele, ...]
    design = np.hstack([design, np.zeros((n_indiv, n_anc))])

    A = np.zeros([1, design.shape[1]])

    A[0, -2] = 1
    A[0, -1] = -1

    df_rls = {"het_pval": [], "coef1": [], "se1": [], "coef2": [], "se2": []}
    pvals = np.zeros(n_snp)

    for snp_i in tqdm(range(n_snp)):
        design[:, -2:] = apa[snp_i, :, :]
        model = sm.OLS(y, design).fit()
        df_rls["het_pval"].append(model.f_test(A).pvalue.item())
        for anc_i in range(2):
            df_rls[f"coef{anc_i + 1}"].append(
                model.params[len(model.params) - 2 + anc_i]
            )
            df_rls[f"se{anc_i + 1}"].append(model.bse[len(model.params) - 2 + anc_i])

    return pd.DataFrame(df_rls)


def test_snp_assoc(apa, y, cov):
    assert apa.shape[1] == 2
    design = sm.add_constant(np.hstack([apa.sum(axis=1)[:, np.newaxis], cov]))
    model = sm.OLS(y, design).fit()
    return model.pvalues[1], model


def simulate_hetero_assoc(
    pfile: str,
    hsq: float,
    causal_snp: str,
    region_window: int = 50,
    n_top_snp: int = 10,
):
    """
    Simulate phenotype based on a region around one causal SNP
    perform association testing for all the SNPs in the region

    Parameters
    ----------

    pfile: pfile
    hsq: heritability of the causal SNPs.
    causal_snp: index of the causal SNP
    region_window: window size in kb
    n_top_snp: top associated SNPs for evaluating heterogeneity
    """
    # format data
    np.random.seed(42)
    geno, df_snp, df_indiv = dapgen.read_pfile(pfile, phase=True, snp_chunk=1024)
    lanc = admix.io.read_lanc(pfile + ".lanc").dask(snp_chunk=1024)
    df_snp_info = pd.read_csv(pfile + ".snp_info", sep="\t").set_index("SNP")
    assert np.all(df_snp_info.index == df_snp.index.values)
    df_snp = pd.merge(df_snp, df_snp_info, left_index=True, right_index=True)

    # determine subset of SNPs
    causal_idx = df_snp.index.get_loc(causal_snp)
    causal_pos = df_snp.loc[causal_snp, "POS"]

    region_snp_idx = np.where(
        (causal_pos - region_window * 1e3 <= df_snp.POS)
        & (df_snp.POS < causal_pos + region_window * 1e3)
    )[0]

    df_tmp = df_snp.iloc[region_snp_idx]
    # filter 0.005 < EUR_FREQ < 0.995 and 0.005 < AFR_FREQ < 0.995
    region_snp_idx = region_snp_idx[
        df_tmp.EUR_FREQ.between(0.005, 0.995) & df_tmp.AFR_FREQ.between(0.005, 0.995)
    ]
    print(len(region_snp_idx))

    geno = geno[region_snp_idx, :, :]
    lanc = lanc[region_snp_idx, :, :]
    df_snp = df_snp.iloc[region_snp_idx, :]

    n_eff_snp = df_snp.shape[0]

    causal_idx = df_snp.index.get_loc(causal_snp)

    # simulate phenotype
    beta = np.zeros((n_eff_snp, 2, 1))  # (n_snp, n_anc, n_sim)
    # use position of SNP to determine the beta sign (to add some randomness)
    beta[causal_idx, :, :] = (-1) ** (df_snp.POS.iloc[causal_idx] % 2)
    sim = simulate_quant_pheno(geno=geno, lanc=lanc, hsq=hsq, beta=beta, n_sim=1)
    pheno = sim["pheno"].flatten()

    df_rls = marginal_het(
        geno=geno,
        lanc=lanc,
        y=pheno,
    )

    df_rls["assoc_pval"] = admix.assoc.marginal(
        geno=geno, lanc=lanc, pheno=pheno, cov=None
    )
    df_rls["causal_snp"] = causal_snp
    df_rls.index = df_snp.index
    df_rls.index.name = "test_snp"
    df_rls = df_rls.sort_values("assoc_pval")
    df_rls["assoc_pval_rank"] = np.arange(len(df_rls))

    # return both the top SNPs and the causal SNP (if that is not within the top)
    rls_snp_idx = df_rls.index[0:n_top_snp].tolist()
    if causal_snp not in rls_snp_idx:
        rls_snp_idx.append(causal_snp)
    return df_rls.loc[rls_snp_idx, :]


def tls(x, y):
    # https://gist.github.com/RyotaBannai/db4d26f7c3c3029e320ae1d28864b36c
    """
    x: np.ndarray
    y: np.ndarray
    """
    import numpy.linalg as la

    X = np.zeros((len(x), 2))
    X[:, 0] = 1.0  # intercept
    X[:, 1] = x
    n = np.array(X).shape[1]

    Z = np.vstack((X.T, y)).T
    U, s, Vt = la.svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = -Vxy / Vyy  # total least squares soln

    Xtyt = -Z.dot(V[:, n:]).dot(V[:, n:].T)
    Xt = Xtyt[:, :n]  # X error
    y_tls = (X + Xt).dot(a_tls)
    fro_norm = la.norm(Xtyt, "fro")
    # slope and intercept
    return [a_tls[1], a_tls[0]]


from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
import numpy as np


def orthoregress(x, y):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c, nan, nan, nan]
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.
    """

    def f(p, x):
        """Basic linear regression 'model' for use with ODR"""
        return (p[0] * x) + p[1]

    linreg = linregress(x, y)
    mod = Model(f)
    dat = Data(x, y)
    od = ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()
    # slope and intercept
    return list(out.beta)


def deming_regression(x, y, sx=None, sy=None, no_intercept=False):
    def no_intercept_func(B, x):
        return B[0] * x

    if no_intercept:
        model = scipy.odr.Model(no_intercept_func)
        odr = scipy.odr.ODR(scipy.odr.RealData(x, y, sx=sx, sy=sy), model, beta0=[1])
        fit = odr.run()
        return fit.beta[0]
    else:
        model = scipy.odr.unilinear
        odr = scipy.odr.ODR(scipy.odr.RealData(x, y, sx=sx, sy=sy), model)
        fit = odr.run()
        return fit.beta[0], fit.beta[1]