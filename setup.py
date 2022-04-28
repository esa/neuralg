"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
"""

from setuptools import setup, find_packages
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="neuralg",
    version=get_version("neuralg/__init__.py"),
    description="Package providing torch-based neural network approximators of linear algebra operations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esa/neuralg",
    author="ESA Advanced Concepts Team",
    author_email="toveag@kth.se",
    install_requires=[
        "matplotlib>=3.3.3",
        "torch==1.9.0",
        "loguru>=0.5.3",
        "dotmap>=1.3.24",
        "toml>=0.10.2",
        "requests >= 2.27.0",
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
    python_requires=">=3.8, <4",
    project_urls={"Source": "https://github.com/esa/neuralg/"},
)
