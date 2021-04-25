import numpy as np
import fire
from admix.data import read_int_mat, write_int_mat, convert_anc_count
import pandas as pd
from utils import read_grm, write_grm


def build_admix_grm(admix_hap, admix_lanc):
    
    assert np.all(admix_hap.shape == admix_lanc.shape)

    admix_geno = convert_anc_count(admix_hap, admix_lanc).astype(float)

    # two ancestries
    index1 = slice(0, n_snp)
    index2 = slice(n_snp, 2 * n_snp)

    K1 = (
        np.dot(admix_geno[:, index1], admix_geno[:, index1].T) / n_snp
        + np.dot(admix_geno[:, index2], admix_geno[:, index2].T) / n_snp
    )

    cross_term = np.dot(admix_geno[:, index1], admix_geno[:, index2].T) / n_snp
    K2 = cross_term + cross_term.T
    return [K1, K2]


# CLI
def cli_build_grm(geno_prefix, out_prefix):
    """
    TODO: extend to three-way admixture
    """
    admix_hap = read_int_mat(geno_prefix + ".hap")
    admix_lanc = read_int_mat(geno_prefix + ".lanc")
    
    n_indiv = admix_hap.shape[0] // 2
    n_snp = admix_hap.shape[1]

    admix_hap = admix_hap.reshape(n_indiv, n_snp * 2)
    admix_lanc = admix_lanc.reshape(n_indiv, n_snp * 2)
    print(f"Number of individual: {n_indiv}, number of SNPs: {n_snp}")
    K1, K2 = build_admix_grm(admix_hap, admix_lanc)
    
    write_grm(out_prefix + "K1", K=K1, df_id=pd.DataFrame({"0": np.arange(n_indiv), "1":np.arange(n_indiv)}), n_snps=np.repeat(n_snp, n_indiv))
    write_grm(out_prefix + "K2", K=K2, df_id=pd.DataFrame({"0": np.arange(n_indiv), "1":np.arange(n_indiv)}), n_snps=np.repeat(n_snp, n_indiv))
    
    with open(out_prefix + "mgrm.txt", 'w') as f:
        f.writelines(["K1\nK2"])
        
if __name__ == "__main__":
    fire.Fire()