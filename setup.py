from setuptools import find_packages
from setuptools import setup

import os


PACKAGE_VERSION = os.environ.get("SHARK_PACKAGE_VERSION") or "0.0.1"

setup(
    name="shark",
    version=f"{PACKAGE_VERSION}",
    description="The SHARK Runner provides inference and training APIs to run deep learning models on Shark Runtime.",
    author="Nod Labs",
    author_email="stdin@nod.com",
    url="https://github.com/NodLabs/dSHARK",
    packages=find_packages(exclude=('dSHARK','examples')),
)
