"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

setup(
    name="neuralg",
    version="0.0.1",
    description="Package providing torch-based neural network approximators of linear algebra operations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esa/neuralg",
    author="ESA Advanced Concepts Team",
    author_email="toveag@kth.se",  #  Should this be me or someone else?
    install_requires=[
        "matplotlib>=3.3.3",
        "torch==1.9.0",
        "loguru>=0.5.3",
        "dotmap>=1.3.24",
        "toml>=0.10.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(include=["neuralg", "neuralg.*"]),
    package_data={
        "neuralg": ["models/saved_models/*.pt"],
    },
    python_requires=">=3.8, <4",
    project_urls={"Source": "https://github.com/esa/neuralg/"},
)
