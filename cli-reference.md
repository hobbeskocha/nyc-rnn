# Jupytext for py to ipynb, and vice versa

jupytext --to .py --output {output-filepath} {notebook-input-filepath}

- jupytext --to .py --output ./src/0_nyc_eda.py src/0_nyc_eda.ipynb
- jupytext --to .py --output ./src/1_nyc_analysis.py src/1_nyc_analysis.ipynb

jupytext --to .ipynb --output {output-filepath} {py-input-filepath}

- jupytext --to .ipynb --output ./src/0_nyc_eda.ipynb src/0_nyc_eda.py
- jupytext --to .ipynb --output ./src/1_nyc_analysis.ipynb src/1_nyc_analysis.py

# nbconvert for ipynb to html

jupyter nbconvert {input-filepath} --output-dir="{filepath}" --to html

- jupyter nbconvert src/0_nyc_eda.ipynb --output-dir="html" --to html
- jupyter nbconvert src/1_nyc_analysis.ipynb --output-dir="html" --to html
