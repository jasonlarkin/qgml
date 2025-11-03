===============================
QGML Installation Guide
===============================

This guide provides comprehensive installation instructions for the Quantum Geometric Machine Learning (QGML) framework.

System Requirements
===================

Hardware Requirements
---------------------

**Minimum Requirements**:
- CPU: 2+ cores, 2.0+ GHz
- RAM: 8 GB
- Storage: 2 GB free space

**Recommended Requirements**:
- CPU: 4+ cores, 3.0+ GHz
- RAM: 16+ GB
- GPU: NVIDIA GPU with CUDA support (optional)
- Storage: 5+ GB free space

Software Requirements
---------------------

**Python Version**:
- Python 3.8 or higher
- Python 3.9-3.11 recommended

**Operating Systems**:
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.14+)
- Windows (10+)

Installation Methods
====================

Method 1: pip Installation (Recommended)
----------------------------------------

**Basic Installation**:
.. code-block:: bash

   pip install qgml

**Development Installation**:
.. code-block:: bash

   pip install -e .

**With GPU Support**:
.. code-block:: bash

   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install QGML
   pip install qgml

Method 2: conda Installation
----------------------------

**Create conda environment**:
.. code-block:: bash

   conda create -n qgml python=3.9
   conda activate qgml

**Install dependencies**:
.. code-block:: bash

   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install numpy scipy matplotlib seaborn
   pip install qgml

Method 3: uv Installation (Fast)
--------------------------------

**Install uv**:
.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

**Install QGML**:
.. code-block:: bash

   uv add qgml

**With development dependencies**:
.. code-block:: bash

   uv add qgml[dev]

Method 4: Source Installation
-----------------------------

**Clone repository**:
.. code-block:: bash

   git clone https://github.com/your-org/qgml.git
   cd qgml

**Install in development mode**:
.. code-block:: bash

   pip install -e .

**Install with all dependencies**:
.. code-block:: bash

   pip install -e ".[dev,test,docs]"

Dependency Management
=====================

Core Dependencies
-----------------

**Required Dependencies**:
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

**Optional Dependencies**:
- JAX >= 0.3.0 (for JAX backend)
- Qiskit >= 0.40.0 (for quantum hardware)
- Seaborn >= 0.11.0 (for advanced plotting)

Development Dependencies
------------------------

**Testing**:
- pytest >= 6.0.0
- pytest-cov >= 3.0.0
- pytest-xdist >= 2.0.0

**Documentation**:
- Sphinx >= 4.0.0
- sphinx-rtd-theme >= 1.0.0
- myst-parser >= 0.17.0

**Code Quality**:
- black >= 22.0.0
- flake8 >= 4.0.0
- mypy >= 0.950

Backend Configuration
=====================

PyTorch Backend (Default)
-------------------------

**Installation**:
.. code-block:: bash

   # CPU only
   pip install torch torchvision torchaudio

   # GPU support (CUDA 11.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # GPU support (CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

**Verification**:
.. code-block:: python

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU count: {torch.cuda.device_count()}")

JAX Backend (Optional)
----------------------

**Installation**:
.. code-block:: bash

   # CPU only
   pip install jax jaxlib

   # GPU support
   pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**Verification**:
.. code-block:: python

   import jax
   print(f"JAX version: {jax.__version__}")
   print(f"JAX devices: {jax.devices()}")

Quantum Hardware Backend (Optional)
-----------------------------------

**Qiskit Installation**:
.. code-block:: bash

   pip install qiskit qiskit-aer

**Verification**:
.. code-block:: python

   import qiskit
   print(f"Qiskit version: {qiskit.__version__}")

Installation Verification
=========================

Basic Verification
------------------

**Test QGML installation**:
.. code-block:: python

   import qgml
   print(f"QGML version: {qgml.__version__}")

**Test backend functionality**:
.. code-block:: python

   from qgml import set_backend, get_backend
   
   # Test PyTorch backend
   set_backend('pytorch')
   print(f"Current backend: {get_backend()}")
   
   # Test JAX backend (if available)
   try:
       set_backend('jax')
       print(f"JAX backend: {get_backend()}")
   except ImportError:
       print("JAX backend not available")

**Test basic functionality**:
.. code-block:: python

   from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
   import torch
   
   # Create trainer
   trainer = SupervisedMatrixTrainer(N=8, D=3)
   
   # Generate test data
   X = torch.randn(50, 3)
   y = torch.randn(50)
   
   # Test training
   history = trainer.fit(X, y, n_epochs=10)
   print("Training completed successfully!")

Comprehensive Verification
--------------------------

**Run test suite**:
.. code-block:: bash

   pytest tests/ -v

**Run specific tests**:
.. code-block:: bash

   # Test core functionality
   pytest tests/test_integration/test_trainer_integration.py -v
   
   # Test backend functionality
   pytest tests/test_backend/ -v
   
   # Test with coverage
   pytest --cov=qgml --cov-report=html

**Run examples**:
.. code-block:: bash

   # Run quickstart example
   python examples/quickstart.py
   
   # Run comprehensive example
   python examples/comprehensive_demo.py

Troubleshooting
===============

Common Installation Issues
--------------------------

**PyTorch Installation Issues**:

*Problem*: CUDA version mismatch
.. code-block:: bash

   # Check CUDA version
   nvidia-smi
   
   # Install matching PyTorch version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

*Problem*: Import errors
.. code-block:: bash

   # Reinstall PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio

**JAX Installation Issues**:

*Problem*: JAX not working with GPU
.. code-block:: bash

   # Reinstall JAX with CUDA support
   pip uninstall jax jaxlib
   pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**QGML Installation Issues**:

*Problem*: Module not found
.. code-block:: bash

   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Reinstall in development mode
   pip install -e .

*Problem*: Backend errors
.. code-block:: python

   # Check backend configuration
   from qgml import get_backend
   print(f"Current backend: {get_backend()}")
   
   # Reset backend
   from qgml import set_backend
   set_backend('pytorch')

Performance Issues
------------------

**Memory Issues**:
.. code-block:: python

   # Check available memory
   import psutil
   print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
   
   # Reduce batch size for large models
   trainer = SupervisedMatrixTrainer(N=8, D=3, batch_size=16)

**GPU Issues**:
.. code-block:: python

   # Check GPU availability
   import torch
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name()}")
       print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
   else:
       print("GPU not available")

Environment Setup
=================

Virtual Environment
-------------------

**Using venv**:
.. code-block:: bash

   python -m venv qgml_env
   source qgml_env/bin/activate  # Linux/macOS
   # qgml_env\Scripts\activate  # Windows
   pip install qgml

**Using conda**:
.. code-block:: bash

   conda create -n qgml python=3.9
   conda activate qgml
   pip install qgml

**Using uv**:
.. code-block:: bash

   uv venv qgml_env
   source qgml_env/bin/activate
   uv add qgml

Environment Variables
--------------------

**CUDA Configuration**:
.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Memory management

**JAX Configuration**:
.. code-block:: bash

   export JAX_PLATFORM_NAME=gpu  # Force GPU usage
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8  # Memory fraction

**QGML Configuration**:
.. code-block:: bash

   export QGML_BACKEND=pytorch  # Default backend
   export QGML_DEVICE=cuda  # Default device

Docker Installation
===================

**Dockerfile**:
.. code-block:: dockerfile

   FROM python:3.9-slim
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       git \
       && rm -rf /var/lib/apt/lists/*
   
   # Install QGML
   COPY . /app
   WORKDIR /app
   RUN pip install -e .
   
   # Set environment variables
   ENV QGML_BACKEND=pytorch
   
   CMD ["python", "-c", "import qgml; print('QGML installed successfully!')"]

**Build and run**:
.. code-block:: bash

   docker build -t qgml .
   docker run -it qgml

**With GPU support**:
.. code-block:: bash

   docker run --gpus all -it qgml

Next Steps
==========

After successful installation:

1. **Read the Quickstart Guide**: :doc:`quickstart`
2. **Explore Basic Concepts**: :doc:`basic_concepts`
3. **Run Examples**: Check the `examples/` directory
4. **Read the API Documentation**: :doc:`../api/core`

See Also
========

* :doc:`quickstart` - Quickstart tutorial
* :doc:`basic_concepts` - Basic QGML concepts
* :doc:`../api/core` - Core API documentation
* :doc:`../testing/index` - Testing documentation
