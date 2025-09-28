# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Custom targets for QGML documentation

# Install documentation dependencies
install-deps:
	pip install -r requirements.txt

# Build HTML documentation
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "HTML documentation built in $(BUILDDIR)/html/"

# Build PDF documentation
pdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@cd "$(BUILDDIR)/latex" && make all-pdf
	@echo "PDF documentation built in $(BUILDDIR)/latex/"

# Clean build directory
clean:
	rm -rf "$(BUILDDIR)"

# Live reload server for development
serve: html
	@echo "Starting live reload server..."
	@cd "$(BUILDDIR)/html" && python -m http.server 8000

# Quick build for development (faster, skips some checks)
quickhtml:
	@$(SPHINXBUILD) -b html -E "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Check documentation for issues
check:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS) $(O)

# Generate API documentation from source code
apidoc:
	sphinx-apidoc -f -o api/ ../qcml/

# Full documentation pipeline
all: clean install-deps apidoc html pdf
	@echo "Complete documentation build finished!"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
