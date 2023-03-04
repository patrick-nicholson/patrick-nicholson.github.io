FROM jupyter/pyspark-notebook

USER ${NB_USER}

RUN mamba install --yes --channel conda-forge \
      psycopg psycopg-c streamz gcc && \
    mamba clean --all -f -y  && \
    python3 -m pip install ray && \
    python3 -m pip install datasketches && \
    python3 -m pip install xgboost && \
    python3 -m pip install jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable spellchecker/main --user && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
