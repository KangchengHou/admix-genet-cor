import numpy as np
import fire

from admix.data import compute_allele_per_anc
import pandas as pd
from utils import read_grm, write_grm
import dask.array as da
from dask.distributed import Client, progress


def build_admix_grm(admix_hap, admix_lanc, n_anc, center=True):

    assert n_anc == 2
    assert np.all(admix_hap.shape == admix_lanc.shape)

    allele_per_anc = compute_allele_per_anc(admix_hap, admix_lanc, n_anc=n_anc).astype(
        float
    )
    n_indiv, n_snp = allele_per_anc.shape[0:2]
    mean_per_anc = allele_per_anc.mean(axis=0)

    a1, a2 = allele_per_anc[:, :, 0], allele_per_anc[:, :, 1]
    if center:
        a1 = a1 - mean_per_anc[:, 0]
        a2 = a2 - mean_per_anc[:, 1]

    K1 = np.dot(a1, a1.T) / n_snp + np.dot(a2, a2.T) / n_snp

    cross_term = np.dot(a1, a2.T) / n_snp
    K2 = cross_term + cross_term.T

    K1, K2 = K1.persist(), K2.persist()
    progress(K1)
    progress(K2)
    return [K1.compute(), K2.compute()]


# def allele_count_per_anc(hap: np.ndarray, lanc: np.ndarray, n_anc: int):
#     """Get allele count per ancestry
#     Parameters
#     ----------
#     hap : np.ndarray
#         haplotype (n_indiv, n_snp, n_anc)
#     lanc : np.ndarray
#         local ancestry (n_indiv, n_snp, n_anc)
#     """
#     assert np.all(hap.shape == lanc.shape), "shape of `hap` and `lanc` are not equal"
#     assert hap.ndim == 3, "`hap` and `lanc` should have three dimension"
#     n_indiv, n_snp, n_haplo = hap.shape
#     assert n_haplo == 2, "`n_haplo` should equal to 2, check your data"
#     geno = np.zeros((n_indiv, n_snp, n_anc), dtype=np.int8)

#     for i_haplo in range(n_haplo):
#         haplo_hap = hap[:, :, i_haplo]
#         haplo_lanc = lanc[:, :, i_haplo]
#         for i_anc in range(n_anc):
#             geno[:, :, i_anc][haplo_lanc == i_anc] += haplo_hap[haplo_lanc == i_anc]

#     return geno


# def legacy_build_admix_grm(admix_hap, admix_lanc, n_anc):

#     assert n_anc == 2
#     assert np.all(admix_hap.shape == admix_lanc.shape)

#     geno = allele_count_per_anc(admix_hap, admix_lanc, n_anc=n_anc).astype(float)
#     n_indiv, n_snp = geno.shape[0], geno.shape[1]

#     K1 = (
#         np.dot(geno[:, :, 0], geno[:, :, 0].T) / n_snp
#         + np.dot(geno[:, :, 1], geno[:, :, 1].T) / n_snp
#     )

#     cross_term = np.dot(geno[:, :, 0], geno[:, :, 1].T) / n_snp
#     K2 = cross_term + cross_term.T
#     return [K1, K2]


# CLI
def cli_build_grm(geno, out_prefix, indiv_subset=None, snp_subset=None, n_anc=2):
    """
    TODO: extend to three-way admixture
    """

    client = Client(n_workers=1, threads_per_worker=2, memory_limit="24GB")

    print("Receiving geno =", geno)
    datasets = []
    if isinstance(geno, list):
        for g in geno:
            datasets.append(g)
    elif isinstance(geno, str):
        datasets.append(geno)
    else:
        raise NotImplementedError
    print(datasets)
    admix_hap = da.hstack([da.from_zarr(d, "hap") for d in datasets])
    admix_lanc = da.hstack([da.from_zarr(d, "lanc") for d in datasets])

    if indiv_subset is not None:
        indiv_subset = np.loadtxt(indiv_subset, dtype=int)
        admix_hap = admix_hap[indiv_subset, :, :]
        admix_lanc = admix_lanc[indiv_subset, :, :]
    if snp_subset is not None:
        snp_subset = np.loadtxt(snp_subset, dtype=int)
        admix_hap = admix_hap[:, snp_subset, :]
        admix_lanc = admix_lanc[:, snp_subset, :]

    print("shape of admix_hap:", admix_hap.shape)
    print("shape of admix_lanc:", admix_lanc.shape)

    assert np.all(admix_hap.shape == admix_lanc.shape)
    n_indiv = admix_hap.shape[0]
    n_snp = admix_hap.shape[1]

    print(f"Number of individual: {n_indiv}, number of SNPs: {n_snp}")
    Ks = build_admix_grm(admix_hap, admix_lanc, n_anc=n_anc)

    names = []
    for i, K in enumerate(Ks):
        name = f"K{i+1}"
        write_grm(
            out_prefix + name,
            K=K,
            df_id=pd.DataFrame({"0": np.arange(n_indiv), "1": np.arange(n_indiv)}),
            n_snps=np.repeat(n_snp, n_indiv),
        )
        names.append(out_prefix + name)

    with open(out_prefix + "mgrm.txt", "w") as f:
        f.writelines("\n".join(names))

    # addition of all GRMs, used for likelihood ratio test
    K_full = sum(Ks)
    write_grm(
        out_prefix + "K_full",
        K=K_full,
        df_id=pd.DataFrame({"0": np.arange(n_indiv), "1": np.arange(n_indiv)}),
        n_snps=np.repeat(n_snp, n_indiv),
    )


if __name__ == "__main__":
    fire.Fire()