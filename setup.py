"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="neuralg",
    version="0.0.1",
    install_requires=[
        "matplotlib>=3.3.3",
        "torch==1.9.0",
        "loguru>=0.5.3",
        "dotmap>=1.3.24",
        "toml>=0.10.2",
    ],
    packages=[
        "neuralg",
        "neuralg.io",
        "neuralg.models",
        "neuralg.ops",
        "neuralg.tests",
        "neuralg.training",
        "neuralg.plots",
        "neuralg.utils",
    ],
    python_requires=">=3.8, <4",
)

