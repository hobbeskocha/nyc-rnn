# Jupytext for py to ipynb, and vice versa 
jupytext --to .py --output {output-filepath} {notebook-input-filepath}
    jupytext --to .py --output ./src/pjm_eda.py src/pjm_eda.ipynb

jupytext --to .ipynb --output {output-filepath} {py-input-filepath}

# nbconvert for ipynb to html
jupyter nbconvert {input-filepath} --output-dir="{filepath}" --to html
    jupyter nbconvert src/pjm_eda.ipynb --output-dir="html" --to html
