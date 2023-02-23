build:
	docker build --rm -t myjupyter .

jupyter: build
	docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work myjupyter

ray: build
	docker run -it --shm-size=4gb --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work myjupyter

postgres: build
	docker-compose up

down:
	docker-compose down
