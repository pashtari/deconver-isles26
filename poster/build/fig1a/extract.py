"""Write fig1a.tex: subfigure (a) of manuscript/architecture.tex as a standalone document."""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "..", "..", "..", "manuscript", "architecture.tex")).read().splitlines()
styles = "\n".join(src[0:59])                    # colours and TikZ styles
body = "\n".join(src[68:148]).rstrip()[:-1]      # the tikzpicture of subfigure (a), without \resizebox's brace
doc = ("\\documentclass[border=4pt]{standalone}\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath}\n\\usepackage{tikz}\n"
       "\\usetikzlibrary{positioning,calc,fit,arrows.meta}\n\\usepackage{xcolor}\n" + styles +
       "\n\\begin{document}\n" + body + "\n\\end{document}\n")
open(os.path.join(here, "fig1a.tex"), "w").write(doc)
