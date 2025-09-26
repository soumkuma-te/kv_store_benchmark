# Key-Value Storage Benchmark: ClickHouse vs Redis

This project provides a comprehensive benchmarking suite to analyze and compare the performance characteristics of ClickHouse and Redis for key-value storage operations.

## Features

- **Comprehensive Schema**: Supports complex data types including strings, multiple integer fields, multiple double fields, and details fields
- **Multiple Benchmark Types**: Single inserts, batch inserts, updates, batch updates, reads, and concurrent operations
- **Performance Metrics**: Measures throughput, latency, percentiles, and success rates
- **Random Data Generation**: Creates realistic test data for accurate benchmarking
- **Detailed Reporting**: Provides formatted results and saves to JSON for further analysis

## Schema

The key-value records include the following fields:
- `id`: Unique identifier (string)
- `source`: Data source (web, mobile, api, batch, streaming)
- `event_type`: Type of event (user_action, system_event, error, warning, info)
- `status`: Record status (active, inactive, pending, completed, failed)
- `string_value_1` through `string_value_5`: Variable-length string fields
- `int64_1` through `int64_5`: Individual 64-bit integer fields
- `dbl_1` through `dbl_5`: Individual double-precision floating-point fields
- `details`: Free-form string for additional information
- `created_at`: Timestamp of record creation

## Prerequisites

### Docker Setup (Recommended)
The easiest way to set up both ClickHouse and Redis is using Docker Compose:

1. **Create data directories**
```bash
mkdir -p data/clickhouse data/clickhouse-logs data/redis
```

2. **Start the databases**
```bash
docker compose up -d
```

3. **Stop the databases**
```bash
docker compose down
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify database connections**
```bash
# Test ClickHouse (should return OK)
curl http://localhost:8123/ping

# Test Redis (requires redis-cli installed locally, or use Docker)
redis-cli ping
# Or using Docker:
docker compose exec redis redis-cli ping

# Check container status
docker compose ps
```

## Docker Compose Management

### Common Commands
```bash
# Start databases in background
docker compose up -d

# Stop databases
docker compose down

# View logs
docker compose logs clickhouse
docker compose logs redis
docker compose logs -f  # Follow all logs

# Restart services
docker compose restart

# Remove everything including volumes
docker compose down -v

# Rebuild and start (if you modify docker-compose.yml)
docker compose up -d --build
```

### Database Connections
- **ClickHouse HTTP Interface**: `http://localhost:8123`
- **ClickHouse Native Interface**: `localhost:9000`
- **Redis**: `localhost:6379`

### Data Storage
- **ClickHouse data**: `./data/clickhouse/`
- **ClickHouse logs**: `./data/clickhouse-logs/`
- **Redis data**: `./data/redis/`

## Usage

### Basic Benchmark
```bash
python key_value_benchmark.py
```

### Custom Configuration
You can modify the benchmark parameters in the `main()` function:

```python
# Example customizations
runner.benchmark_single_inserts(record_count=2000)
runner.benchmark_batch_inserts(record_count=10000, batch_size=200)
runner.benchmark_reads(record_count=1500)
runner.benchmark_concurrent_operations(record_count=800, max_workers=20)
```

## Benchmark Types

### 1. Single Insert Operations
- Measures individual record insertion performance
- Tests both ClickHouse and Redis sequentially
- Provides per-operation timing statistics

### 2. Batch Insert Operations
- Tests bulk insertion capabilities
- Configurable batch sizes
- Compares batch efficiency between databases

### 3. Update Operations
- Measures single record update performance
- Tests modification of all fields for existing records
- Compares update efficiency between databases

### 4. Batch Update Operations
- Tests bulk update capabilities
- Configurable batch sizes for updates
- Measures update throughput for large datasets

### 5. Read Operations
- Benchmarks record retrieval by ID
- Tests query performance for both systems
- Measures success rates and timing

### 6. Concurrent Operations
- Multi-threaded insertion testing
- Configurable worker thread count
- Tests system behavior under concurrent load
- Uses one ClickHouse client per worker thread (initialized once per thread for efficiency)
- Automatic cleanup of thread-local resources

## Performance Metrics

For each benchmark, the following metrics are collected:

- **Total Time**: Complete operation duration
- **Operations per Second**: Throughput measurement
- **Average Time per Record**: Mean operation latency
- **P95 Time**: 95th percentile latency
- **Min/Max Time**: Fastest and slowest operations
- **Success Rate**: Percentage of successful operations

## Output

### Console Output
```
BENCHMARK RESULTS
================================================================================

SINGLE_INSERT OPERATIONS:
--------------------------------------------------------------------------------
Database: CLICKHOUSE
  Records: 1,000
  Total Time: 2.456s
  Ops/Second: 407.1
  Avg Time/Record: 2.456ms
  P95 Time: 4.123ms
  Success Rate: 100.0%

Database: REDIS
  Records: 1,000
  Total Time: 0.891s
  Ops/Second: 1,122.3
  Avg Time/Record: 0.891ms
  P95 Time: 1.234ms
  Success Rate: 100.0%
```

### JSON Results File
Results are automatically saved to `benchmark_results_YYYYMMDD_HHMMSS.json` for further analysis.

## Database Connection Configuration

### ClickHouse
```python
clickhouse = ClickHouseStorage(
    host='localhost',
    port=8123,
    database='default'
)
```

### Redis
```python
redis_storage = RedisStorage(
    host='localhost',
    port=6379,
    db=0
)
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify databases are running: `docker compose ps`
   - Check container logs: `docker compose logs clickhouse` or `docker compose logs redis`
   - Restart services: `docker compose restart`
   - Check port accessibility: `telnet localhost 8123` / `telnet localhost 6379`

2. **Permission Errors**
   - Ensure ClickHouse has write permissions for table creation
   - Verify Redis allows connections from localhost
   - Check data directory permissions: `ls -la data/`

3. **ClickHouse Concurrent Session Issues**
   - The benchmark uses ThreadPoolExecutor initializer for efficient client management
   - One ClickHouse client per worker thread (reused for all tasks on that thread)
   - ClickHouse doesn't support concurrent queries in the same session
   - Automatic cleanup of thread-local clients when threads finish

4. **Docker Issues**
   - Ensure Docker and Docker Compose are installed and running
   - Check if ports are already in use: `lsof -i :8123` / `lsof -i :6379`
   - Clean up old containers: `docker compose down && docker compose up -d`
   - Free up disk space if containers fail to start

5. **Memory Issues**
   - Reduce `record_count` for large datasets
   - Adjust `batch_size` for batch operations
   - Monitor system memory usage during benchmarks

### Performance Tuning

1. **ClickHouse Optimization**
   - Uses `ReplacingMergeTree` engine for efficient updates
   - Efficient concurrent operations with one client per worker thread
   - Thread-local client reuse minimizes connection overhead
   - Adjust `max_insert_block_size` setting
   - Consider different table engines (ReplicatedMergeTree, etc.)
   - Tune `ORDER BY` clause for your access patterns

2. **Redis Optimization**
   - Enable Redis persistence if needed (`appendonly yes`)
   - Adjust `maxmemory` and `maxmemory-policy`
   - Consider Redis clustering for larger datasets

## Extending the Benchmark

### Adding New Storage Systems
1. Implement the storage interface with `insert_record`, `batch_insert_records`, and `get_record` methods
2. Add the new storage to the `BenchmarkRunner`
3. Update the benchmark methods to include the new system

### Custom Data Schemas
Modify the `KeyValueRecord` dataclass and update the storage implementations accordingly.

### Additional Metrics
Extend the `BenchmarkResult` class to capture additional performance characteristics.

