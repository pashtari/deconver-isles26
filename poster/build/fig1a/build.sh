#!/usr/bin/env bash
# Compile subfigure (a) of manuscript/architecture.tex as a standalone figure -> figs/fig1a.svg
cd "$(dirname "$0")"
python3 extract.py && pdflatex -interaction=nonstopmode -halt-on-error fig1a.tex > fig1a.log \
    && pdftocairo -svg fig1a.pdf ../../figs/fig1a.svg && rm -f fig1a.aux fig1a.log && echo "figs/fig1a.svg written"
