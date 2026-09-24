#!/usr/bin/env bash
# Render poster.html to poster.pdf (A0 portrait, 841 x 1189 mm) with headless Chrome.
cd "$(dirname "$0")/.."
google-chrome --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/poster.pdf" "file://$PWD/poster.html" 2>/dev/null
pdfinfo poster.pdf | grep -E "Pages|Page size"
