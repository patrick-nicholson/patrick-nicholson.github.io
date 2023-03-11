FROM jupyter/pyspark-notebook

USER root

RUN mamba install --yes --channel conda-forge \
      psycopg psycopg-c streamz gcc jupyterlab-spellchecker && \
    mamba clean --all -f -y  && \
    python3 -m pip install datasketches && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
