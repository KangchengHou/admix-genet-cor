from distutils.version import LooseVersion
from io import open

import setuptools
from setuptools import setup

# Added support for environment markers in install_requires.
if LooseVersion(setuptools.__version__) < "36.2":
    raise ImportError("setuptools>=36.2 is required")

setup(
    name="admix-genet-cor",
    version="0.1",
    description="Toolbox for analyzing genetics data from admixed population",
    author="Kangcheng Hou",
    author_email="kangchenghou@gmail.com",
    packages=["admix_genet_cor"],
    setup_requires=["numpy>=1.10"],
)