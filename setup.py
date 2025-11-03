from setuptools import setup, find_packages

setup(
    name="qgml",
    version="0.1.0",
    description="Quantum Geometric Machine Learning with dual JAX/PyTorch backends",
    author="Jason Larkin",
    author_email="jasonlarkin84@gmail.com",
    url="https://github.com/jasonlarkin/qgml",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "jax": ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "pytorch": ["torch>=2.0.0", "torchvision>=0.15.0"],
        "quantum": ["qiskit>=0.45.0"],
        "full": ["jax>=0.4.0", "jaxlib>=0.4.0", "qiskit>=0.45.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
