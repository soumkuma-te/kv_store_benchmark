# Key-Value Storage Benchmark Analysis Report

## Executive Summary

This report analyzes the performance characteristics of ClickHouse and Redis for key-value storage operations based on a comprehensive benchmark conducted on September 29, 2025. The benchmark used a "heavy" configuration testing various operation types including single inserts, batch operations, reads, updates, and concurrent operations.

## Benchmark Configuration

- **Test Scenario**: Heavy workload
- **Single Operations**: 5,000 records each
- **Batch Operations**: 50,000 records with batch size of 500
- **Read Tests**: 5,000 records
- **Concurrent Tests**: 2,000 records with 20 workers
- **Index Granularity**: 1024 (ClickHouse)

## Performance Results Summary

### 1. Single Insert Operations

| Database | Records | Ops/Second | Avg Time/Record | P95 Time | Success Rate |
|----------|---------|------------|-----------------|----------|--------------|
| **Redis** | 5,000 | **3,632.30** | **0.28ms** | **0.41ms** | 100% |
| ClickHouse | 5,000 | 103.31 | 9.68ms | 12.82ms | 100% |

**Winner: Redis** (35x faster)

### 2. Batch Insert Operations

| Database | Records | Ops/Second | Avg Time/Batch | P95 Time | Success Rate |
|----------|---------|------------|----------------|----------|--------------|
| **Redis** | 50,000 | **40,457.58** | **12.36ms** | **13.90ms** | 100% |
| ClickHouse | 50,000 | 28,892.28 | 17.31ms | 24.49ms | 100% |

**Winner: Redis** (1.4x faster)

### 3. Read Operations

| Database | Records | Ops/Second | Avg Time/Record | P95 Time | Success Rate |
|----------|---------|------------|-----------------|----------|--------------|
| **Redis** | 5,000 | **7,605.86** | **0.13ms** | **0.17ms** | 100% |
| ClickHouse | 5,000 | 163.05 | 6.13ms | 7.62ms | 100% |

**Winner: Redis** (47x faster)

### 4. Single Update Operations

| Database | Records | Ops/Second | Avg Time/Record | P95 Time | Success Rate |
|----------|---------|------------|-----------------|----------|--------------|
| **Redis** | 5,000 | **4,698.49** | **0.21ms** | **0.27ms** | 100% |
| ClickHouse | 5,000 | 96.77 | 10.33ms | 13.41ms | 100% |

**Winner: Redis** (49x faster)

### 5. Batch Update Operations

| Database | Records | Ops/Second | Avg Time/Batch | P95 Time | Success Rate |
|----------|---------|------------|----------------|----------|--------------|
| **Redis** | 50,000 | **42,144.58** | **11.86ms** | **12.83ms** | 100% |
| ClickHouse | 50,000 | 28,423.80 | 17.59ms | 25.81ms | 100% |

**Winner: Redis** (1.5x faster)

### 6. Concurrent Insert Operations

| Database | Records | Ops/Second | Avg Time/Record | P95 Time | Success Rate |
|----------|---------|------------|-----------------|----------|--------------|
| **Redis** | 2,000 | **446.19** | **2.24ms** | **3.56ms** | 100% |
| ClickHouse | 2,000 | 11.75 | 85.14ms | 120.71ms | 100% |

**Winner: Redis** (38x faster)

## Key Findings

### Performance Patterns

1. **Redis Dominance**: Redis outperformed ClickHouse in all tested scenarios
2. **Single vs Batch Operations**: The performance gap narrows significantly in batch operations
3. **Concurrent Performance**: ClickHouse shows significant degradation under concurrent load
4. **Consistency**: Both systems achieved 100% success rates across all tests

### Performance Multipliers (Redis vs ClickHouse)

- Single Inserts: **35x faster**
- Batch Inserts: **1.4x faster**
- Reads: **47x faster**
- Single Updates: **49x faster**
- Batch Updates: **1.5x faster**
- Concurrent Inserts: **38x faster**

## Redis vs ClickHouse: Comprehensive Comparison

### Redis Advantages

#### Performance
- **Ultra-fast operations**: Sub-millisecond response times for most operations
- **Excellent concurrent performance**: Maintains low latency under concurrent load
- **Memory-based storage**: Direct memory access provides superior speed
- **Simple data model**: Key-value operations are optimized for speed

#### Simplicity
- **Easy to deploy and manage**: Minimal configuration required
- **Simple data model**: Straightforward key-value paradigm
- **Rich data types**: Supports strings, hashes, lists, sets, sorted sets
- **Active community**: Extensive documentation and community support

#### Scalability
- **Horizontal scaling**: Redis Cluster for distributed setups
- **Replication**: Master-slave replication for high availability
- **Pub/Sub capabilities**: Built-in messaging functionality

### Redis Disadvantages

#### Limitations
- **Memory constraints**: Limited by available RAM
- **Data persistence concerns**: Primary storage is volatile memory with risk of data loss
- **Durability challenges**: Requires careful configuration (RDB/AOF) to ensure data persistence
- **Limited querying**: No SQL-like query capabilities
- **Cost at scale**: Memory costs can be prohibitive for large datasets

#### Operational
- **Memory management**: Requires careful memory planning and monitoring
- **Backup complexity**: Snapshotting large datasets can be challenging
- **Single-threaded**: Limited to single-threaded operations (though Redis 6+ has some multi-threading)

### ClickHouse Advantages

#### Analytics & Querying
- **SQL support**: Full SQL query capabilities with advanced analytics functions
- **Column-oriented storage**: Optimized for analytical workloads
- **Data compression**: Excellent compression ratios reduce storage costs
- **Complex aggregations**: Built for OLAP workloads and complex analytics

#### Scalability & Storage
- **Unlimited storage**: Disk-based storage not limited by RAM
- **Horizontal scaling**: Built-in sharding and replication
- **Data durability**: Persistent storage with ACID properties and guaranteed data integrity
- **Zero data loss**: Immediate disk persistence ensures no data loss on system failures
- **Time-series optimization**: Excellent for time-series data and logs

#### Enterprise Features
- **Data types**: Rich data type support including arrays, nested structures
- **Materialized views**: Pre-computed aggregations for faster queries
- **Partitioning**: Automatic data partitioning for performance
- **Integration**: Excellent integration with data engineering tools

### ClickHouse Disadvantages

#### Performance
- **Higher latency**: Slower for simple key-value operations
- **Concurrent limitations**: Performance degrades under high concurrent load
- **Complex queries overhead**: Even simple queries have parsing overhead
- **Write amplification**: MergeTree engine can cause write amplification

#### Complexity
- **Configuration complexity**: Requires more tuning and optimization
- **Learning curve**: SQL knowledge and ClickHouse-specific optimizations needed
- **Operational overhead**: More complex deployment and maintenance

## Recommendations

### Choose Redis When:
- **Ultra-low latency** is critical (< 1ms response times)
- **Simple key-value operations** are primary use case
- **High concurrent load** with simple operations
- **Caching layer** or session storage requirements
- **Real-time applications** requiring immediate responses
- **Limited data size** that fits comfortably in memory
- **Acceptable data loss risk** or when data can be easily regenerated
- **Temporary/ephemeral data** storage requirements

### Choose ClickHouse When:
- **Complex analytical queries** are required
- **Large datasets** that exceed memory capacity
- **Time-series data** and log analytics
- **Data warehouse** and OLAP requirements
- **Cost optimization** for large-scale storage
- **Integration** with existing SQL-based tools and workflows
- **Data durability and persistence** are critical business requirements
- **Zero data loss tolerance** and ACID compliance needed
- **Long-term data retention** and historical analysis required

### Hybrid Approach
Consider using both systems together:
- **Redis**: For hot data, caching, and real-time operations
- **ClickHouse**: For analytical queries, historical data, and reporting
- **Data pipeline**: Stream data through Redis for real-time processing, then archive to ClickHouse for analytics

## Conclusion

This benchmark clearly demonstrates that **Redis excels in operational scenarios** requiring fast, simple key-value operations, while **ClickHouse is optimized for analytical workloads**. However, the performance characteristics vary significantly between operation types:

### Performance Summary by Operation Type

**Single Operations (35-49x Redis advantage):**
- Single inserts, updates, and reads show Redis's overwhelming superiority
- ClickHouse's query parsing and transaction overhead becomes apparent
- Redis's in-memory architecture provides unmatched low-latency performance

**Batch Operations (1.4-1.5x Redis advantage):**
- Batch inserts (40,458 vs 28,892 ops/sec) and updates (42,145 vs 28,424 ops/sec) show much closer competition
- ClickHouse's columnar storage and batch processing capabilities significantly reduce the performance gap
- The smaller difference suggests ClickHouse is well-suited for bulk data operations typical in analytical workloads

### Architectural Implications

The choice between them should be driven by specific use case requirements:

- For **operational systems** prioritizing speed and simplicity over durability: **Redis**
- For **analytical systems** requiring complex queries, large-scale storage, and guaranteed data persistence: **ClickHouse**
- For **bulk processing workloads**: ClickHouse becomes much more competitive, especially when SQL capabilities are needed
- For **comprehensive solutions**: Consider a hybrid architecture leveraging both systems' strengths

The dramatic difference between single operations (35-49x faster) and batch operations (1.4-1.5x faster) reveals each system's architectural strengths: Redis dominates in real-time, low-latency scenarios where some data loss is acceptable, while ClickHouse's batch-oriented design, SQL capabilities, storage efficiency, and guaranteed durability make it ideal for analytical and reporting workloads where data integrity is paramount and bulk operations are common.
