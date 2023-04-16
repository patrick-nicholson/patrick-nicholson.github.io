FROM continuumio/miniconda3

USER root

RUN conda install --yes --channel conda-forge mamba && \
    conda clean --all -f -y  && \
    mamba install --yes --channel conda-forge \
        jupyter \
        jupyterlab \
        ipywidgets \
        jupyterlab-spellchecker \
        matplotlib-base \
        pandas \
        pyarrow \
        scipy \
        scikit-learn \
        dask \
        streamz \
        tqdm \
        statsmodels \
        plotly_express \
        plotly-orca \
        psycopg \
        psycopg-c && \
    mamba clean --all -f -y && \
    python3 -m pip install 'polars[pandas,numpy,pyarrow]'


# FROM jupyter/datascience-notebook

# USER root

# RUN mamba install --yes --channel conda-forge \
#     jupyterlab-spellchecker streamz gcc polars && \
#     mamba clean --all -f -y  && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"
