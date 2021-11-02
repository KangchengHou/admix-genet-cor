"""
Functions for examining locus-level heterogeneity
"""
import numpy as np
import statsmodels.api as sm

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


def test_snp_assoc(apa, y, cov):
    assert apa.shape[1] == 2
    design = sm.add_constant(np.hstack([apa.sum(axis=1)[:, np.newaxis], cov]))
    model = sm.OLS(y, design).fit()
    return model.pvalues[1], model
