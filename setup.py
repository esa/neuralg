"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="neuralg",
    packages=[
        "neuralg",
        "neuralg.io",
        "neuralg.models",
        "neuralg.ops",
        "neuralg.tests",
        "neuralg.training",
        "neuralg.plots",
        "neuralg.utils",
        "neuralg.scripts",
    ],
)
