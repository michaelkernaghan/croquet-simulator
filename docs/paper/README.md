# Croquet AI Research Paper

arXiv-style LaTeX paper documenting the DQN croquet AI research.

## Compilation

### Windows (with MiKTeX or TeX Live)
```bash
pdflatex croquet_ai_paper.tex
pdflatex croquet_ai_paper.tex  # Run twice for references
```

### With Make (if available)
```bash
make
```

### Clean build files
```bash
make clean
# or manually delete: *.aux *.log *.out *.toc *.bbl *.blg
```

## Structure

- `croquet_ai_paper.tex` - Main paper source
- `figures/` - Place training curves and diagrams here (TODO)

## Updating

The paper has TODO sections for:
- Training results (Section 6)
- Training curve figures
- Quantitative evaluation metrics

Update these as training completes and results become available.
