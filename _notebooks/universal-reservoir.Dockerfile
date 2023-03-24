FROM jupyter/datascience-notebook

USER root

RUN mamba install --yes --channel conda-forge \
    jupyterlab-spellchecker streamz gcc && \
    mamba clean --all -f -y  && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
