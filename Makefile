all: ps

ps:
	latex proiect
	latex proiect
	bibtex proiect
	latex proiect
	dvips -o proiect.ps proiect.dvi

pdf: ps
	ps2pdf proiect.ps

clean:
	rm -f *.aux *.fdb* *.fls *.lof *.log *.lot *.out *.synctex* *.toc
