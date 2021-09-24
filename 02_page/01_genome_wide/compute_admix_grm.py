from dask.distributed import Client, progress

client = Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit="45GB")

import xarray as xr
import numpy as np
import admix
import matplotlib.pyplot as plt
from utils import *
import dask
import dask.array as da
import fire


def compute_admix_grm(chrom):
    dset = load_hm3(chrom=chrom)

    admix.tools.admix_grm(dset, center=True)
    dset["A1"] = (
        ("indiv", "indiv"),
        dset["admix_grm_K1"].data + dset["admix_grm_K2"].data,
    )
    dset["A2"] = (
        ("indiv", "indiv"),
        dset["admix_grm_K12"].data + dset["admix_grm_K12"].data.T,
    )

    rls = dask.persist(
        dset["admix_grm_K1"], dset["admix_grm_K2"], dset["admix_grm_K12"]
    )
    progress(rls)
    K1, K2, K12 = rls

    np.save(f"out/admix_grm/K1.chr{chrom}.npy", K1.compute())
    np.save(f"out/admix_grm/K2.chr{chrom}.npy", K2.compute())
    np.save(f"out/admix_grm/K12.chr{chrom}.npy", K12.compute())


if __name__ == "__main__":
    fire.Fire(compute_admix_grm)
