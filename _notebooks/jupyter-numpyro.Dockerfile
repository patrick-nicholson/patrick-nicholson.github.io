FROM continuumio/miniconda3

USER root

RUN conda install --yes --channel conda-forge mamba && \
    conda clean --all -f -y  && \
    mamba install --yes --channel conda-forge \
        jupyterlab \
        jupyterlab-spellchecker \
        matplotlib-base \
        pandas \
        scipy \
        "numpyro[cpu]" && \
    mamba clean --all -f -y
