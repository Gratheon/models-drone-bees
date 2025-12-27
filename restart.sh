#!/bin/bash
cd /www/models-drone-bees/

COMPOSE_PROJECT_NAME=gratheon docker-compose down

COMPOSE_PROJECT_NAME=gratheon docker-compose up -d --build

