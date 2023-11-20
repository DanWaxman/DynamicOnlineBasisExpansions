from setuptools import setup, find_packages

setup(
    name="DOEBE",
    version="0.0.1",
    url="https://github.com/DanWaxman/DynamicOnlineBasisExpansions",
    author="Dan Waxman",
    author_email="daniel.waxman@stonybrook.edu",
    description="Code for our paper Dynamic Online Ensembles of Basis Expansions",
    packages=find_packages(),
    install_requires=[
        "jax >= 0.4.20",
        "jaxlib >= 0.4.20",
        "objax >= 1.8.0",
        "libsvmdata >= 0.4.1",
        "scikit-learn >= 1.3.2",
        "tensorflow-probability >= 0.22.1",
        "tqdm >= 4.66.1",
    ],
)
