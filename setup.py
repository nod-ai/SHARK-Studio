from setuptools import find_packages
from setuptools import setup


setup(
    name="shark",
    version="0.0.1",
    description="The Shark Runner provides inference and training APIs to run deep learning models on Shark Runtime.",
    author="Nod Labs",
    author_email="stdin@nod.com",
    url="https://github.com/NodLabs/dSHARK",
    packages=find_packages(exclude=('dSHARK','examples')),
)
