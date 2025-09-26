#!/usr/bin/env python3
"""
Example usage script for the key-value benchmark suite.
Demonstrates how to run custom benchmarks and analyze results.
"""

import json
import time
from datetime import datetime

from key_value_benchmark import (
    ClickHouseStorage, RedisStorage, BenchmarkRunner, 
    DataGenerator, KeyValueRecord
)
from benchmark_config import get_database_configs

def example_custom_benchmark():
    """Example of running a custom benchmark scenario"""
    print("Running Custom Benchmark Example")
    print("=" * 40)
    
    # Get database configurations
    clickhouse_config, redis_config = get_database_configs()
    
    # Initialize storage systems
    try:
        clickhouse = ClickHouseStorage(
            host=clickhouse_config.host,
            port=clickhouse_config.port,
            **clickhouse_config.additional_params
        )
        redis_storage = RedisStorage(
            host=redis_config.host,
            port=redis_config.port,
            **redis_config.additional_params
        )
        print("✓ Connected to both databases")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return
    
    # Create benchmark runner
    runner = BenchmarkRunner(clickhouse, redis_storage)
    
    # Custom test: Small records vs Large records performance
    print("\n1. Testing small vs large record performance...")
    
    # Generate small records (minimal data)
    small_records = []
    for i in range(100):
        record = DataGenerator.generate_record()
        record.details = "small"  # Small details field
        # Set small values for int64 fields
        record.int64_1 = 1
        record.int64_2 = 2
        record.int64_3 = 3
        record.int64_4 = 1
        record.int64_5 = 2
        # Set small values for double fields
        record.dbl_1 = 1.0
        record.dbl_2 = 2.0
        record.dbl_3 = 1.0
        record.dbl_4 = 2.0
        record.dbl_5 = 1.0
        small_records.append(record)
    
    # Generate large records (lots of data)
    large_records = []
    for i in range(100):
        record = DataGenerator.generate_record()
        record.details = "x" * 1000  # Large details field
        # Set large values for int64 fields
        record.int64_1 = 100000
        record.int64_2 = 200000
        record.int64_3 = 300000
        record.int64_4 = 400000
        record.int64_5 = 500000
        # Set large values for double fields
        record.dbl_1 = 100.0
        record.dbl_2 = 200.0
        record.dbl_3 = 300.0
        record.dbl_4 = 400.0
        record.dbl_5 = 500.0
        large_records.append(record)
    
    # Test small records
    print("  Testing small records...")
    start_time = time.time()
    for record in small_records:
        clickhouse.insert_record(record)
    ch_small_time = time.time() - start_time
    
    start_time = time.time()
    for record in small_records:
        redis_storage.insert_record(record)
    redis_small_time = time.time() - start_time
    
    # Test large records
    print("  Testing large records...")
    start_time = time.time()
    for record in large_records:
        clickhouse.insert_record(record)
    ch_large_time = time.time() - start_time
    
    start_time = time.time()
    for record in large_records:
        redis_storage.insert_record(record)
    redis_large_time = time.time() - start_time
    
    # Display results
    print(f"\n  Small Records (100 records):")
    print(f"    ClickHouse: {ch_small_time:.3f}s ({100/ch_small_time:.1f} ops/s)")
    print(f"    Redis:      {redis_small_time:.3f}s ({100/redis_small_time:.1f} ops/s)")
    print(f"  Large Records (100 records):")
    print(f"    ClickHouse: {ch_large_time:.3f}s ({100/ch_large_time:.1f} ops/s)")
    print(f"    Redis:      {redis_large_time:.3f}s ({100/redis_large_time:.1f} ops/s)")
    
    # Cleanup
    runner.cleanup()

def example_analyze_results():
    """Example of analyzing benchmark results from a file"""
    print("\n\nAnalyzing Benchmark Results")
    print("=" * 40)
    
    # This would analyze results from a previous benchmark run
    # For demonstration, we'll show how to work with the results structure
    
    sample_results = [
        {
            "operation": "single_insert",
            "database": "clickhouse",
            "records_count": 1000,
            "total_time": 2.456,
            "avg_time_per_record": 0.002456,
            "ops_per_second": 407.1,
            "min_time": 0.001,
            "max_time": 0.015,
            "p95_time": 0.008,
            "success_rate": 1.0
        },
        {
            "operation": "single_insert", 
            "database": "redis",
            "records_count": 1000,
            "total_time": 0.891,
            "avg_time_per_record": 0.000891,
            "ops_per_second": 1122.3,
            "min_time": 0.0005,
            "max_time": 0.003,
            "p95_time": 0.002,
            "success_rate": 1.0
        }
    ]
    
    # Analyze performance differences
    for operation in set(r["operation"] for r in sample_results):
        op_results = [r for r in sample_results if r["operation"] == operation]
        
        if len(op_results) >= 2:
            ch_result = next((r for r in op_results if r["database"] == "clickhouse"), None)
            redis_result = next((r for r in op_results if r["database"] == "redis"), None)
            
            if ch_result and redis_result:
                print(f"\n{operation.upper()} Analysis:")
                
                # Throughput comparison
                throughput_ratio = redis_result["ops_per_second"] / ch_result["ops_per_second"]
                print(f"  Throughput: Redis is {throughput_ratio:.1f}x faster than ClickHouse")
                
                # Latency comparison
                latency_ratio = ch_result["avg_time_per_record"] / redis_result["avg_time_per_record"]
                print(f"  Latency: ClickHouse is {latency_ratio:.1f}x slower than Redis")
                
                # P95 latency comparison
                p95_ratio = ch_result["p95_time"] / redis_result["p95_time"]
                print(f"  P95 Latency: ClickHouse is {p95_ratio:.1f}x slower than Redis")
                
                # Recommendations
                print(f"  Recommendation:")
                if throughput_ratio > 2:
                    print(f"    Redis shows significantly better performance for {operation}")
                    print(f"    Consider Redis for high-frequency operations")
                else:
                    print(f"    Performance is relatively similar")
                    print(f"    Choice depends on other factors (persistence, querying, etc.)")

def example_update_benchmark():
    """Example of running update performance benchmarks"""
    print("\n\nUpdate Performance Benchmark Example")
    print("=" * 40)
    
    # Get database configurations
    clickhouse_config, redis_config = get_database_configs()
    
    # Initialize storage systems
    try:
        clickhouse = ClickHouseStorage(
            host=clickhouse_config.host,
            port=clickhouse_config.port,
            **clickhouse_config.additional_params
        )
        redis_storage = RedisStorage(
            host=redis_config.host,
            port=redis_config.port,
            **redis_config.additional_params
        )
        print("✓ Connected to both databases")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return
    
    # Create benchmark runner
    runner = BenchmarkRunner(clickhouse, redis_storage)
    
    print("\n1. Testing single record updates...")
    runner.benchmark_updates(record_count=100)
    
    print("\n2. Testing batch updates...")
    runner.benchmark_batch_updates(record_count=500, batch_size=50)
    
    # Clean up
    print("Cleaning up test data...")
    runner.cleanup()
    
    # Print results
    print("\nUpdate Performance Results:")
    print("-" * 40)
    
    for result in runner.results:
        if "update" in result.operation:
            print(f"\n{result.operation.upper()} - {result.database.upper()}:")
            print(f"  Records: {result.records_count:,}")
            print(f"  Total time: {result.total_time:.3f}s")
            print(f"  Ops/second: {result.ops_per_second:.1f}")
            print(f"  Avg time/record: {result.avg_time_per_record*1000:.2f}ms")
            print(f"  P95 time: {result.p95_time*1000:.2f}ms")
            print(f"  Success rate: {result.success_rate*100:.1f}%")

def example_data_generation():
    """Example of custom data generation"""
    print("\n\nCustom Data Generation Example")
    print("=" * 40)
    
    # Generate records with specific patterns
    print("Generating 5 sample records...")
    
    for i in range(5):
        record = DataGenerator.generate_record()
        print(f"\nRecord {i+1}:")
        print(f"  ID: {record.id}")
        print(f"  Source: {record.source}")
        print(f"  Event Type: {record.event_type}")
        print(f"  Status: {record.status}")
        print(f"  String Values: {[record.string_value_1[:10] + '...' if len(record.string_value_1) > 10 else record.string_value_1]}")
        print(f"  Int64 Fields: [{record.int64_1}, {record.int64_2}, {record.int64_3}, {record.int64_4}, {record.int64_5}]")
        print(f"  Double Fields: [{round(record.dbl_1, 2)}, {round(record.dbl_2, 2)}, {round(record.dbl_3, 2)}, {round(record.dbl_4, 2)}, {round(record.dbl_5, 2)}]")
        print(f"  Details: {record.details[:20]}...")

def main():
    """Run all examples"""
    print("Key-Value Benchmark Suite - Examples")
    print("=" * 50)
    
    # Run examples
    example_data_generation()
    example_custom_benchmark()
    example_update_benchmark()
    example_analyze_results()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the full benchmark:")
    print("  python key_value_benchmark.py --scenario light")
    print("  python key_value_benchmark.py --scenario default")
    print("  python key_value_benchmark.py --scenario heavy")

if __name__ == "__main__":
    main()
