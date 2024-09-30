.PHONY: all
all:
	latexmk -pdf -jobname=output -output-directory=cache -aux-directory=cache -pdflatex="pdflatex -interaction=nonstopmode" -use-make main.tex

.PHONY: clean
clean:
	latexmk -c
	rm -f *.bbl *.run.xml *.bcf *.blg *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
