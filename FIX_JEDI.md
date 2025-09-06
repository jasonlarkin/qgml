# Fix Jedi Error

The jedi module error happens because jedi isn't in the same Python environment as Jupyter.

## Quick Fix:

```bash
cd /mnt/c/Users/jason/Documents/qcml_new/qgml

# Option 1: Install in uv environment
uv run pip install jedi

# Option 2: If that doesn't work, find Jupyter's Python and install there
python -c "import jupyter_client; import sys; print(sys.executable)" | xargs -I {} {} -m pip install jedi

# Option 3: Nuclear option - reinstall jupyter with jedi
uv run pip install --force-reinstall jupyter jedi
```

After installing, restart Jupyter.

