"""
Setup configuration for the Crowd-Certain package.

This module handles the installation and packaging of the Crowd-Certain library,
which provides tools for crowd-sourced label aggregation with uncertainty estimation
and confidence scoring.
"""

import os
import codecs
from setuptools import find_packages, setup

# Get the absolute path to the directory containing setup.py
here = os.path.abspath(os.path.dirname(__file__))

# Read the README file
with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with codecs.open(os.path.join(here, "crowd_certain", "config", "requirements.txt"), encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
PACKAGE_NAME = "crowd-certain"
VERSION = "1.0.0"
AUTHOR = "Artin Majdi"
AUTHOR_EMAIL = "msm2024@gmail.com"
DESCRIPTION = "A comprehensive framework for crowd-sourced label aggregation with uncertainty estimation and confidence scoring"
URL = "https://github.com/artinmajdi/taxonomy"
LICENSE = "Apache License 2.0"

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
]

# Package data
PACKAGE_DATA = {
    "crowd_certain": [
        "config/*.json",
        "config/requirements.txt",
        "config/environment.yml",
        "docs/**/*",
        "scripts/**/*",
        "utilities/**/*",
        "datasets/**/*",
        "notebooks/**/*",
        "outputs/**/*",
    ]
}

# Development requirements
DEV_REQUIREMENTS = [
    "pytest>=8.3.5",
    "black>=25.1.0",
    "isort>=6.0.1",
    "flake8>=7.1.2",
    "mypy>=1.15.0",
    "sphinx>=8.2.3",
    "sphinx-rtd-theme>=3.0.2",
    "sphinx-autodoc-typehints>=3.1.0",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": [
            "sphinx>=8.2.3",
            "sphinx-rtd-theme>=3.0.2",
            "sphinx-autodoc-typehints>=3.1.0",
        ],
        "storage": [
            "h5py>=3.7.0",
        ],
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "crowd-dashboard=crowd_certain.run_streamlit:main",
        ],
    },
    keywords=[
        "crowd-sourcing",
        "label-aggregation",
        "uncertainty-estimation",
        "confidence-scoring",
        "machine-learning",
        "data-science",
        "hdf5-storage",
    ],
    project_urls={
        "Bug Tracker": f"{URL}/issues",
        "Documentation": f"{URL}/blob/main/crowd_certain/docs/README.md",
        "Source Code": URL,
    },
)
