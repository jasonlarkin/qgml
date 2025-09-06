from setuptools import setup, find_packages

setup(
    name="qgml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ],
    python_requires=">=3.7",
) 