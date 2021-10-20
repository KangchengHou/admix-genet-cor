import admix
import numpy as np
import dask
from dask.distributed import Client
import fire
import xarray as xr

client = Client(processes=False, threads_per_worker=1, n_workers=1, memory_limit='28GB')

def prepare_dataset(n_indiv, n_snp, out, n_anc=2, mosaic_size=100, anc_props=[0.3, 0.7], seed=1234):
    print(f"Receive anc_props={anc_props}")
    np.random.seed(seed)
    dset = admix.simulate.admix_geno(n_indiv=n_indiv, n_snp=n_snp, n_anc=n_anc, mosaic_size=mosaic_size, anc_props = anc_props)
    admix.tools.admix_grm(dset, center=True)
    dset["A1"] = (("indiv", "indiv"), dset["admix_grm_K1"].data + dset["admix_grm_K2"].data)
    dset["A2"] = (("indiv", "indiv"), dset["admix_grm_K12"].data + dset["admix_grm_K12"].data.T)
    
    dset.to_zarr(out, mode="w")

def simulate_pheno(dset, cor, out, n_causal=None, n_sim=50, seed=1234):
    """
    Heritability is fixed to 0.5 for now, the parameter `cor` controls 
    the genetic correlation, with cor = 0 as completely uncorrelated and
    cor = 1 as perfectly correlated
    """
    np.random.seed(seed)
    dset = xr.open_zarr(dset)
    sim = admix.simulate.continuous_pheno(dset, var_g=1.0, gamma=cor, var_e=1.0,
                                          n_causal=n_causal, n_sim=n_sim)
#    print(dset["allele_per_anc"].values[0:20, 0:20, :])
#     grms = {"A1": dset["A1"].data,
#             "A2": dset["A2"].data}
#     grms = dask.persist(grms)[0]

    np.save(out, sim["pheno"])
    
def estimate(dset, pheno, out, i_sim, method="HE"):
    """
    dset: path to the zarr dataset
    pheno: (n_indiv, n_sim) numpy matrix
    """
    assert method == "HE"
    dset = xr.open_zarr(dset)
    pheno = np.load(pheno)
    
    rls = admix.estimate.admix_gen_cor(dset=dset, pheno=pheno[:, i_sim])[0]
    text = "estimate\n" + str(rls[0]) + '\nvarcov\n' + str(rls[1])     
    with open(out, 'w') as f:
        f.write(text)
        
if __name__ == "__main__":
    fire.Fire()