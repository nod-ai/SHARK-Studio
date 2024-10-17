from setuptools import find_packages
from setuptools import setup

import os
import glob

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PACKAGE_VERSION = os.environ.get("SHARK_PACKAGE_VERSION") or "2.0.0"
backend_deps = []

setup(
    name="nodai-SHARK",
    version=f"{PACKAGE_VERSION}",
    description="SHARK provides a High Performance Machine Learning Framework",
    author="nod.ai",
    author_email="stdin@nod.ai",
    url="https://nod.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Code": "https://github.com/nod-ai/SHARK",
        "Bug Tracker": "https://github.com/nod-ai/SHARK-Studio/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("examples")),
    python_requires=">=3.9",
    data_files=glob.glob("apps/stable_diffusion/resources/**"),
    install_requires=[
        "numpy",
        "PyYAML",
    ]
)
