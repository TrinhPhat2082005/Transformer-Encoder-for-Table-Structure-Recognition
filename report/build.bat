@echo off
echo Dang build PDF...
latexmk -pdf main.tex
echo.
echo Da build xong. Dang don dep file rac...
latexmk -c
del *.synctex.gz
echo.
echo Hoan tat! Chi con lai file PDF.
