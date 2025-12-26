start:
	docker compose -f docker-compose.dev.yml up --build

start-prod:
	docker compose -f docker-compose.prod.yml up --build

stop:
	docker compose -f docker-compose.dev.yml down

stop-prod:
	docker compose -f docker-compose.prod.yml down

run-local:
	python3 server.py

test:
	@echo "Testing server with GET request..."
	@curl -s http://localhost:8751 | head -10

logs:
	docker compose -f docker-compose.dev.yml logs -f

