import numpy as np
import fire
from admix.data import read_int_mat, write_int_mat, allele_count_per_anc
import pandas as pd
from utils import read_grm, write_grm
import zarr


def build_admix_grm(admix_hap, admix_lanc, n_anc):

    assert n_anc == 2
    assert np.all(admix_hap.shape == admix_lanc.shape)

    geno = allele_count_per_anc(admix_hap, admix_lanc, n_anc=n_anc).astype(float)
    n_indiv, n_snp = geno.shape[0], geno.shape[1]

    K1 = (
        np.dot(geno[:, :, 0], geno[:, :, 0].T) / n_snp
        + np.dot(geno[:, :, 1], geno[:, :, 1].T) / n_snp
    )

    cross_term = np.dot(geno[:, :, 0], geno[:, :, 1].T) / n_snp
    K2 = cross_term + cross_term.T
    return [K1, K2]


# CLI
def cli_build_grm(geno, out_prefix, n_anc=2):
    """
    TODO: extend to three-way admixture
    """

    print("Receiving geno =", geno)
    datasets = []
    if isinstance(geno, list):
        for g in geno:
            datasets.append(zarr.load(g))
    elif isinstance(geno, str):
        datasets.append(zarr.load(geno))
    else:
        raise NotImplementedError

    admix_hap = np.hstack([d["hap"] for d in datasets])
    admix_lanc = np.hstack([d["lanc"] for d in datasets])
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