import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import itertools
import admix
from tqdm import tqdm
import os
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import glob
import string
from scipy import stats

def read_estimate(snpset, ukb_trait_list, page_trait_list):
    dict_loglik = dict()
    dict_nindiv = dict()
    rho_list = np.linspace(0, 1, 21)
    xs = np.linspace(0, 1, 1001)

    for study in ["ukb", "page"]:
        trait_list = ukb_trait_list if study == "ukb" else page_trait_list
        for trait in trait_list:
            if study == "ukb":
                est_dir = (
                    "/u/home/k/kangchen/project-UKBB/UKB-ADMIXED/02-genet-cor/"
                    f"out/gcta-estimate/{trait}-sample10pc-{snpset}"
                )
            else:
                est_dir = (
                    "/u/project/pasaniuc/kangchen/2021-admix-corr/experiments/"
                    "03-page-genome-wide-profile-likelihood-new/out/gcta-estimate/"
                    f"{trait}-sample10pc-{snpset}"
                )
            try:
                loglik_list = [
                    admix.tools.gcta.read_reml(
                        os.path.join(est_dir, f"rho{int(rho * 100)}")
                    )["loglik"]
                    for rho in rho_list
                ]
                nindiv = admix.tools.gcta.read_reml(os.path.join(est_dir, "rho100"))[
                    "n"
                ]
                cs = CubicSpline(rho_list, loglik_list)
                ll = cs(xs)
                dict_loglik[(study, trait)] = ll
                dict_nindiv[(study, trait)] = nindiv
            except ValueError as err:
                dict_loglik[(study, trait)] = None
                dict_nindiv[(study, trait)] = None
                print(trait, study, snpset, err)
    return dict_loglik, dict_nindiv