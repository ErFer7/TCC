.PHONY: all clean main.pdf

all: main.pdf

main.pdf: main.tex
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -rf *.aux *.bbl *.bcf *.blg *.idx *.log *.run.xml *.toc chapters/*.aux