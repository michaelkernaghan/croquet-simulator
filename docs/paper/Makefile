# Makefile for Croquet AI paper

PAPER = croquet_ai_paper

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz

cleanall: clean
	rm -f $(PAPER).pdf

.PHONY: all clean cleanall
