install:
	@make build
	@make up
up:
	docker compose up -d
build:
	docker compose build
remake:
	@make destroy
	@make install
down:
	docker compose down --remove-orphans
restart:
	@make down
	@make up
destroy:
	docker compose down --rmi all --volumes --remove-orphans