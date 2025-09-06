# Quick Start - Jupyter with QGML

## Option 1: Just install and run (simplest)

```bash
cd /mnt/c/Users/jason/Documents/qcml_new/qgml
uv pip install jupyter jupyterlab ipykernel
uv run jupyter lab examples/qrc_mnist_qgml_analysis.ipynb
```

## Option 2: If uv pip doesn't work, use pip directly

```bash
cd /mnt/c/Users/jason/Documents/qcml_new/qgml
python -m pip install jupyter jupyterlab ipykernel
jupyter lab examples/qrc_mnist_qgml_analysis.ipynb
```

## Option 3: If you already have a venv activated

```bash
cd /mnt/c/Users/jason/Documents/qcml_new/qgml
pip install jupyter jupyterlab ipykernel
jupyter lab examples/qrc_mnist_qgml_analysis.ipynb
```

That's it. No kernel registration needed - just run the notebook.

