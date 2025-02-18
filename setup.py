# setup.py
from setuptools import setup, find_packages

setup(
    name="reduxcache",
    version="0.1.0",
    description="Caching utility for reduxprior pipeline embeddings in Hugging Face Diffusers",
    author="sahir2k",
    url="https://github.com/sahir2k/reduxcache",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers",
        "Pillow",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
