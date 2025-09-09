#!/bin/bash

# Compile LaTeX document with BibTeX
echo "Compiling QCML technical document with BibTeX..."

# First pass: pdflatex
echo "Running first pdflatex pass..."
pdflatex working_paper.tex

# Run BibTeX
echo "Running BibTeX..."
bibtex working_paper

# Second pass: pdflatex (to resolve citations)
echo "Running second pdflatex pass..."
pdflatex working_paper.tex

# Third pass: pdflatex (to resolve cross-references)
echo "Running third pdflatex pass..."
pdflatex working_paper.tex

echo "Compilation complete! Check working_paper.pdf"
echo "This version includes all citations and should produce a complete bibliography."
echo ""
echo "Files generated:"
echo "- working_paper.pdf (main document)"
echo "- working_paper.aux (auxiliary file)"
echo "- working_paper.log (compilation log)"
echo "- working_paper.out (hyperref output)"
echo "- working_paper.toc (table of contents)"
echo "- working_paper.bbl (bibliography)"
echo "- working_paper.blg (bibliography log)"
