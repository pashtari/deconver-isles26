#!/usr/bin/env bash
# Report page count and the bottom margin of page 1 (helper while adjusting the layout).
cd "$(dirname "$0")/.."
pdftoppm -png -r 30 -f 1 -l 1 poster.pdf /tmp/pm >/dev/null 2>&1
python3 -c "
from PIL import Image; im=Image.open('/tmp/pm-1.png').convert('L'); w,h=im.size
rows=[y for y in range(h) if min(im.crop((0,y,w,y+1)).getdata())<250]; print(f'page 1 bottom margin: {(h-1-rows[-1])/h*1189:.0f} mm')"
