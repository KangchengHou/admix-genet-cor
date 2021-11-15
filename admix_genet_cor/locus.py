"""
Functions for examining locus-level heterogeneity
"""
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import admix
import pandas as pd

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

    z_list = np.zeros(n_snp)
    # for each snp, calculate the covariance
    transform_vec = np.array([[1], [-1]])
    for snp_i in tqdm(range(n_snp)):
        cov_inv = np.linalg.inv(np.cov(apa[snp_i, :, :], rowvar=False))
        # expectation and variance
        ex = cov_inv @ apa[snp_i, :, :].T @ m
#         z_list[snp_i] = transform_vec.T @ ex
        z_list[snp_i] = (
            transform_vec.T @ ex / np.sqrt(transform_vec.T @ cov_inv @ transform_vec)
        )
    return z_list

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
    
    if cov is None:
        design = np.ones((n_indiv, 1))
    else:
        design = np.hstack([np.ones((n_indiv, 1)), cov])
    design = np.hstack([design, np.zeros((n_indiv, n_anc))])
    
    A = np.zeros([1, design.shape[1]])
    A[0, 1] = 1
    A[0, 2] = -1
    
    df_rls = {
        "het_pval": [],
        "coef1": [],
        "se1": [],
        "coef2": [],
        "se2": []
    }
    pvals = np.zeros(n_snp)

    for snp_i in tqdm(range(n_snp)):
        design[:, -2:] = apa[snp_i, :, :]
        model = sm.OLS(y, design).fit()
        df_rls['het_pval'].append(model.f_test(A).pvalue.item())
        for anc_i in range(2):
            df_rls[f"coef{anc_i + 1 }"].append(model.params[anc_i + 1])
            df_rls[f"se{anc_i + 1}"].append(model.bse[anc_i + 1])
        

    return pd.DataFrame(df_rls)
    
def test_snp_assoc(apa, y, cov):
    assert apa.shape[1] == 2
    design = sm.add_constant(np.hstack([apa.sum(axis=1)[:, np.newaxis], cov]))
    model = sm.OLS(y, design).fit()
    return model.pvalues[1], model
