colima:
	colima start --cpu 4 --memory 6

build:
	docker build --rm -t myjupyter .

jupyter: colima build
	docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work myjupyter

postgres: colima build
	docker-compose up

down: colima
	docker-compose down
	docker volume prune -f
	docker network prune -f

stop: colima down
	colima stop