#!/bin/bash
# Setup script for Jupyter with QGML

cd /mnt/c/Users/jason/Documents/qcml_new/qgml

echo "Installing Jupyter dependencies..."
uv pip install jupyter jupyterlab ipykernel

echo ""
echo "Setup complete!"
echo ""
echo "To start Jupyter Lab, run:"
echo "  uv run jupyter lab examples/qrc_mnist_qgml_analysis.ipynb"
echo ""
echo "Optional: Register kernel for Jupyter:"
echo "  uv run python -m ipykernel install --user --name=qgml --display-name 'Python (QGML)'"

