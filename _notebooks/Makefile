colima:
	colima start --cpu 4 --memory 6

# jupyter: colima
# 	docker build --rm -t jupyter_default -f jupyter-default.Dockerfile .
# 	docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work jupyter_default

jupyter: colima
	docker build --rm -t jupyter_default -f jupyter-default.Dockerfile .
	docker run -it --rm -p 8888:8888 -v "${PWD}":/opt/notebooks jupyter_default /bin/bash -c "\
		mkdir -p /opt/notebooks && \
		jupyter-lab --notebook-dir=/opt/notebooks --ip='*' --port=8888 --allow-root"

numpyro: colima
	docker build --rm -t jupyter_numpyro -f jupyter-numpyro.Dockerfile .
	docker run -it --rm -p 8888:8888 -v "${PWD}":/opt/notebooks jupyter_numpyro /bin/bash -c "\
		mkdir -p /opt/notebooks && \
		jupyter-lab --notebook-dir=/opt/notebooks --ip='*' --port=8888 --allow-root"

spark: colima
	docker build --rm -t jupyter_spark -f jupyter-spark.Dockerfile .
	docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work jupyter_spark

postgres: colima
	docker build --rm -t jupyter_postgres -f jupyter-postgres.Dockerfile .
	docker-compose -f compose-postgres.yaml up

down: colima
	docker-compose -f compose-postgres.yaml down
	docker volume prune -f
	docker network prune -f

stop: colima down
	colima stop