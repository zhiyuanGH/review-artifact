"""Minimal editable installation for the anonymous framework package."""

from setuptools import find_packages, setup


setup(
    name="mobench-framework",
    version="0.1.0",
    description="Mobile agent workflow planning, deployment, and execution framework",
    packages=["MOBench", *["MOBench." + package for package in find_packages()]],
    package_dir={"MOBench": "."},
    package_data={"MOBench": ["data/*.json", "configs/*.example.json"]},
    python_requires=">=3.10",
)
