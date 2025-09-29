# Makefile for Key-Value Benchmark Suite

.PHONY: help install setup clean benchmark-light benchmark-default benchmark-heavy example start-db stop-db

# Default target
help:
	@echo "Key-Value Benchmark Suite"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo "  install         - Install Python dependencies"
	@echo "  start-db       - Start ClickHouse and Redis using Docker Compose"
	@echo "  setup          - Set up ClickHouse and Redis databases"
	@echo "  example        - Run example usage script"
	@echo "  benchmark-light - Run light benchmark scenario"
	@echo "  benchmark-default - Run default benchmark scenario"
	@echo "  benchmark-heavy - Run heavy benchmark scenario"
	@echo "  stop-db        - Stop database containers"
	@echo "  clean          - Clean up benchmark results, data directories, and Python cache"
	@echo "  help           - Show this help message"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Set up databases
setup:
	@echo "Setting up databases..."
	python setup_databases.py
	@echo "✓ Database setup completed"

# Start databases using Docker Compose
start-db:
	@echo "Creating data directories..."
	mkdir -p data/clickhouse data/clickhouse-logs data/redis
	@echo "Starting ClickHouse and Redis containers..."
	docker compose up -d
	@echo "✓ Database containers started"
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "✓ Services should be ready"

# Run light benchmark
benchmark-light:
	@echo "Running light benchmark scenario..."
	python key_value_benchmark.py --scenario light

# Run default benchmark
benchmark-default:
	@echo "Running default benchmark scenario..."
	python key_value_benchmark.py --scenario default

# Run heavy benchmark
benchmark-heavy:
	@echo "Running heavy benchmark scenario..."
	python key_value_benchmark.py --scenario heavy

# Run example
example:
	@echo "Running example usage script..."
	python example_usage.py

# Clean up results and data
clean:
	@echo "Cleaning up benchmark results..."
	rm -f benchmark_results_*.json
	@echo "Cleaning up data directories..."
	rm -rf data/
	@echo "Cleaning up Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleanup completed"

# Stop database containers
stop-db:
	@echo "Stopping database containers..."
	docker compose down
	@echo "✓ Database containers stopped"

# Full setup and test
test: install setup benchmark-light
	@echo "✓ Full test completed successfully"
