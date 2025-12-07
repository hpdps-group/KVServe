"""
Setup script for kvserve package
"""

from setuptools import setup, find_packages

setup(
    name="kvserve",
    version="0.1.0",
    description="PD Separation Engine based on vLLM",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ray",
        "vllm>=0.10.1",
        "torch",
        "transformers",
    ],
)


