# Fix Jupyter - Direct Commands

## If uv is causing problems, use pip directly:

```bash
cd /mnt/c/Users/jason/Documents/qcml_new/qgml

# Activate your venv if you have one, or use system python
python3 -m pip install --user jupyter jupyterlab ipykernel

# Then just run:
jupyter lab examples/qrc_mnist_qgml_analysis.ipynb
```

## Or if you have a venv already:

```bash
cd /mnt/c/Users/jason/Documents/qcml_new/qgml
source .venv/bin/activate  # or whatever your venv path is
pip install jupyter jupyterlab ipykernel
jupyter lab examples/qrc_mnist_qgml_analysis.ipynb
```

## Skip all the setup - just open the notebook file directly

The notebook file is at: `examples/qrc_mnist_qgml_analysis.ipynb`

You can open it in VS Code or any editor that supports notebooks, or just install jupyter with pip and run it.

