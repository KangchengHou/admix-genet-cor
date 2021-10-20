import fire
import numpy as np
from utils import read_grm, write_grm, trace_mul
from os.path import join
from scipy import linalg
import pandas as pd
from scipy import linalg

# load K1, K2
def main(grm_dir, phen, out):
    K1, df_id1, n_snps1 = read_grm(join(grm_dir, "K1"))
    K2, df_id2, n_snps2 = read_grm(join(grm_dir, "K2"))
    phe = pd.read_csv(phen, sep='\t', header=None)
    n_indiv = phe.shape[0]
    n_sim = phe.shape[1] - 2
    quad_form_func = lambda x, A: np.dot(np.dot(x.T, A), x)
    
    design = np.array(
        [
            [trace_mul(K1, K1), trace_mul(K1, K2), np.trace(K1)],
            [trace_mul(K2, K1), trace_mul(K2, K2), np.trace(K2)],
            [np.trace(K1), np.trace(K2), n_indiv],
        ]
    )

    df_rls = []
    print(f"Number of simulations: {n_sim}")
    for i_sim in range(n_sim):
        y=phe.loc[:, i_sim + 2].values
        response = np.array([quad_form_func(y, K1), quad_form_func(y, K2), np.sum(y * y)])
        rls = linalg.solve(design, response)
        df_rls.append(rls)
    
    df_rls = np.vstack(df_rls)
    df_rls = pd.DataFrame(df_rls, columns=["var_g", "gamma", "var_e"])
    
    print("Mean:")
    print(df_rls.mean(axis=0))
    print("SD:")
    print(df_rls.std(axis=0))
    
    df_rls.to_csv(out, index=False)

if __name__ == "__main__":
    fire.Fire(main)