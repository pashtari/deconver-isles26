# Poster

`poster.pdf` is the A0 portrait poster (841 x 1189 mm) presented at the SWITCH+ 2026 workshop
(MICCAI 2026, Strasbourg, September 27). It is rendered from `poster.html` with headless
Chrome; the Inter and Inter Display fonts must be installed, and LaTeX (pdflatex) is needed
only to regenerate the equations and Fig. 1a.

```bash
python poster/build/make_chart.py            # figs/dice_vs_params.svg (Tables 1-2 of the paper)
python poster/build/make_diagrams.py         # figs/mixer.svg, figs/blocks4.svg (Fig. 1b)
bash   poster/build/fig1a/build.sh           # figs/fig1a.svg from manuscript/architecture.tex
bash   poster/build/equations/build.sh       # figs/eq_*.svg
python poster/build/make_qualitative.py --stage select  --data_dir /path/to/ISLES26   # candidate cases
python poster/build/make_qualitative.py --stage predict --data_dir /path/to/ISLES26   # fold-0 checkpoints, GPU
python poster/build/make_qualitative.py --stage table                                  # per-case Dice incl. ensemble
python poster/build/make_qualitative.py --stage draw --cases id1,id2,id3               # figs/qualitative.png
bash   poster/build/render.sh                # poster.pdf
```

All numbers on the poster come from the paper. The qualitative panels are predictions of the
released fold-0 checkpoints (and their SegResNet + Deconver ensemble) on fold-0 validation
cases, cached under `build/cache/`. The QR code links to https://github.com/pashtari/deconver.
