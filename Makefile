all: clean pdf

pdf:
	latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex --shell-escape proiect

clean:
	rm -f *.acn *.acr *.alg *.aux *.bbl *.blg *.dvi *.fdb_latexmk *.fls *.glg *.glo *.gls *.idx *.ilg *.ind *.lof *.log *.lot *.nav *.out *.ps *.snm *.synctex* *.toc *.vrb *.xdv
