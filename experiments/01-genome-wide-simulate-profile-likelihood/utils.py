import numpy as np
from admix.data import convert_anc_count

def read_grm(file_prefix):

    from numpy import (
        asarray,
        float32,
        float64,
        fromfile,
        int64,
        tril,
        tril_indices_from,
        zeros,
    )
    from pandas import read_csv

    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    df_id = read_csv(id_file, sep="\t", header=None, names=["sample_0", "sample_1"])
    n = df_id.shape[0]
    k = asarray(fromfile(bin_file, dtype=float32), float64)
    n_snps = asarray(fromfile(N_file, dtype=float32), int64)

    K = zeros((n, n))
    K[tril_indices_from(K)] = k
    K = K + tril(K, -1).T

    return (K, df_id, n_snps)


def write_grm(file_prefix, K, df_id, n_snps):

    from numpy import (
        float32,
        tril_indices_from,
    )

    bin_file = file_prefix + ".grm.bin"
    N_file = file_prefix + ".grm.N.bin"
    id_file = file_prefix + ".grm.id"

    # id
    df_id.to_csv(id_file, sep="\t", header=None, index=False)
    # bin
    K[tril_indices_from(K)].astype(float32).tofile(bin_file)
    # N
    n_snps.astype(float32).tofile(N_file)
    

def calculate_admix_grm(admix_geno):
    admix_geno = admix_geno.astype(float)
    n_indiv = admix_geno.shape[0]
    n_snp = admix_geno.shape[1] // 2

    # two ancestries
    index1 = slice(0, n_snp)
    index2 = slice(n_snp, 2 * n_snp)

    K1 = (
        np.dot(admix_geno[:, index1], admix_geno[:, index1].T) / n_snp
        + np.dot(admix_geno[:, index2], admix_geno[:, index2].T) / n_snp
    )

    cross_term = np.dot(admix_geno[:, index1], admix_geno[:, index2].T) / n_snp
    K2 = cross_term + cross_term.T
    return K1, K2


def trace_mul(a, b):
    """
    Trace of two matrix inner product
    """
    assert np.all(a.shape == b.shape)
    return np.sum(a.flatten() * b.flatten())