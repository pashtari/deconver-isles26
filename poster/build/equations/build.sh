#!/usr/bin/env bash
# Typeset the poster equations (Eqs. 1 and 3 of the paper) -> figs/eq_*.svg
cd "$(dirname "$0")"
for n in eq_block eq_ndc_problem eq_ndc; do
    pdflatex -interaction=nonstopmode -halt-on-error $n.tex > $n.log && pdftocairo -svg $n.pdf ../../figs/$n.svg && rm -f $n.aux $n.log
done
echo "figs/eq_*.svg written"
