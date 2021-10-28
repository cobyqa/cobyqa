# Makefile for logo creation
#

LC = xelatex
LFLAGS = -file-line-error -interaction=nonstopmode -shell-escape

.PHONY: main clean

main: logo.tex
	$(LC) $(LFLAGS) $^

clean:
	-rm -f *.aux *.log *.pdf *.svg
