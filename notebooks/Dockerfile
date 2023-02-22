FROM jupyter/pyspark-notebook

USER root

RUN mamba install --yes --channel conda-forge \
    psycopg psycopg-c &&\
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
