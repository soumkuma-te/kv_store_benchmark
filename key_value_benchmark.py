#!/usr/bin/env python3
"""
Key-Value Storage Benchmark: ClickHouse vs Redis
Analyzes performance characteristics of ClickHouse and Redis for key-value operations.
"""

import json
import random
import string
import time
import uuid
import argparse
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import statistics
import concurrent.futures
from datetime import datetime

# Database imports
import clickhouse_connect
import redis
import pandas as pd
import numpy as np

# Local imports
from benchmark_config import (
    get_benchmark_config, 
    get_database_configs,
    BenchmarkConfig
)

# Thread-local storage for ClickHouse clients
thread_local_data = threading.local()

def init_thread_clickhouse_client(host, port, connection_params, table_name):
    """Initialize a ClickHouse client for the current thread"""
    try:
        thread_local_data.clickhouse_client = clickhouse_connect.get_client(
            host=host, 
            port=port, 
            **connection_params
        )
        thread_local_data.table_name = table_name
    except Exception as e:
        print(f"Error initializing thread ClickHouse client: {e}")
        thread_local_data.clickhouse_client = None
        thread_local_data.table_name = table_name

def cleanup_thread_clickhouse_client():
    """Clean up the ClickHouse client for the current thread"""
    try:
        if hasattr(thread_local_data, 'clickhouse_client') and thread_local_data.clickhouse_client:
            thread_local_data.clickhouse_client.close()
    except Exception as e:
        print(f"Error cleaning up thread ClickHouse client: {e}")

@dataclass
class KeyValueRecord:
    """Schema for key-value storage records"""
    id: str
    source: str
    event_type: str
    status: str
    string_value_1: str
    string_value_2: str
    string_value_3: str
    string_value_4: str
    string_value_5: str
    int64_1: int
    int64_2: int
    int64_3: int
    int64_4: int
    int64_5: int
    dbl_1: float
    dbl_2: float
    dbl_3: float
    dbl_4: float
    dbl_5: float
    details: str
    created_at: datetime

@dataclass
class BenchmarkResult:
    """Results from benchmark operations"""
    operation: str
    database: str
    records_count: int
    total_time: float
    avg_time_per_record: float
    ops_per_second: float
    min_time: float
    max_time: float
    p95_time: float
    success_rate: float

class DataGenerator:
    """Generates random test data for benchmarking"""
    
    SOURCES = ["web", "mobile", "api", "batch", "streaming"]
    EVENT_TYPES = ["user_action", "system_event", "error", "warning", "info"]
    STATUSES = ["active", "inactive", "pending", "completed", "failed"]
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate random string of specified length"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_int64() -> int:
        """Generate random 64-bit integer"""
        return random.randint(1, 1000000)
    
    @staticmethod
    def random_double() -> float:
        """Generate random double"""
        return round(random.uniform(0.0, 1000.0), 3)
    
    @classmethod
    def generate_record(cls) -> KeyValueRecord:
        """Generate a single random record"""
        return KeyValueRecord(
            id=str(uuid.uuid4()),
            source=random.choice(cls.SOURCES),
            event_type=random.choice(cls.EVENT_TYPES),
            status=random.choice(cls.STATUSES),
            string_value_1=cls.random_string(20),
            string_value_2=cls.random_string(15),
            string_value_3=cls.random_string(25),
            string_value_4=cls.random_string(32),
            string_value_5=cls.random_string(64),
            int64_1=cls.random_int64(),
            int64_2=cls.random_int64(),
            int64_3=cls.random_int64(),
            int64_4=cls.random_int64(),
            int64_5=cls.random_int64(),
            dbl_1=cls.random_double(),
            dbl_2=cls.random_double(),
            dbl_3=cls.random_double(),
            dbl_4=cls.random_double(),
            dbl_5=cls.random_double(),
            details=cls.random_string(1024),
            created_at=datetime.now()
        )
    
    @classmethod
    def generate_records(cls, count: int) -> List[KeyValueRecord]:
        """Generate multiple random records"""
        return [cls.generate_record() for _ in range(count)]

class ClickHouseStorage:
    """ClickHouse storage implementation"""
    
    def __init__(self, host: str = 'localhost', port: int = 8123, **kwargs):
        # Set default credentials if not provided
        if 'username' not in kwargs:
            kwargs['username'] = 'benchmark_user'
        if 'password' not in kwargs:
            kwargs['password'] = 'benchmark_password'
        if 'database' not in kwargs:
            kwargs['database'] = 'benchmark_db'
        
        # Store connection parameters for creating new clients in concurrent operations
        self.host = host
        self.port = port
        self.connection_params = kwargs.copy()
            
        self.client = clickhouse_connect.get_client(host=host, port=port, **kwargs)
        self.table_name = 'key_value_store'
        self._create_table()
    
    def _create_table(self):
        """Create the key-value table if it doesn't exist"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id String,
            source String,
            event_type String,
            status String,
            string_value_1 String,
            string_value_2 String,
            string_value_3 String,
            string_value_4 String,
            string_value_5 String,
            int64_1 Int64,
            int64_2 Int64,
            int64_3 Int64,
            int64_4 Int64,
            int64_5 Int64,
            dbl_1 Float64,
            dbl_2 Float64,
            dbl_3 Float64,
            dbl_4 Float64,
            dbl_5 Float64,
            details String,
            created_at DateTime
        ) ENGINE = ReplacingMergeTree()
        ORDER BY id
        SETTINGS index_granularity = 1024
        """
        self.client.command(create_table_sql)
    
    def insert_record(self, record: KeyValueRecord) -> float:
        """Insert a single record and return operation time"""
        start_time = time.time()
        
        data = [[
            record.id,
            record.source,
            record.event_type,
            record.status,
            record.string_value_1,
            record.string_value_2,
            record.string_value_3,
            record.string_value_4,
            record.string_value_5,
            record.int64_1,
            record.int64_2,
            record.int64_3,
            record.int64_4,
            record.int64_5,
            record.dbl_1,
            record.dbl_2,
            record.dbl_3,
            record.dbl_4,
            record.dbl_5,
            record.details,
            record.created_at
        ]]
        
        self.client.insert(self.table_name, data)
        return time.time() - start_time
    
    def batch_insert_records(self, records: List[KeyValueRecord]) -> float:
        """Insert multiple records in batch and return operation time"""
        start_time = time.time()
        
        data = []
        for record in records:
            data.append([
                record.id,
                record.source,
                record.event_type,
                record.status,
                record.string_value_1,
                record.string_value_2,
                record.string_value_3,
                record.string_value_4,
                record.string_value_5,
                record.int64_1,
                record.int64_2,
                record.int64_3,
                record.int64_4,
                record.int64_5,
                record.dbl_1,
                record.dbl_2,
                record.dbl_3,
                record.dbl_4,
                record.dbl_5,
                record.details,
                record.created_at
            ])
        
        self.client.insert(self.table_name, data)
        return time.time() - start_time
    
    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get a record by ID and return operation time"""
        start_time = time.time()
        
        query = f"SELECT * FROM {self.table_name} WHERE id = %(id)s LIMIT 1"
        result = self.client.query(query, parameters={'id': record_id})
        
        operation_time = time.time() - start_time
        
        if result.result_rows:
            return result.result_rows[0], operation_time
        return None, operation_time
    
    def update_record(self, record: KeyValueRecord) -> float:
        """Update a single record and return operation time"""
        # With ReplacingMergeTree, we can just insert again with same ID
        return self.insert_record(record)
    
    def batch_update_records(self, records: List[KeyValueRecord]) -> float:
        """Update multiple records and return operation time"""
        # With ReplacingMergeTree, we can just insert again with same IDs
        return self.batch_insert_records(records)
    
    def create_new_client(self):
        """Create a new ClickHouse client with same connection parameters for concurrent operations"""
        return clickhouse_connect.get_client(
            host=self.host, 
            port=self.port, 
            **self.connection_params
        )
    
    def clear_table(self):
        """Clear all data from the table"""
        self.client.command(f"TRUNCATE TABLE {self.table_name}")

class RedisStorage:
    """Redis storage implementation"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, **kwargs):
        self.client = redis.Redis(host=host, port=port, **kwargs)
        self.key_prefix = 'kvstore:'
    
    def insert_record(self, record: KeyValueRecord) -> float:
        """Insert a single record and return operation time"""
        start_time = time.time()
        
        # Convert record to JSON for storage
        record_dict = asdict(record)
        record_dict['created_at'] = record.created_at.isoformat()
        
        key = f"{self.key_prefix}{record.id}"
        self.client.set(key, json.dumps(record_dict))
        
        return time.time() - start_time
    
    def batch_insert_records(self, records: List[KeyValueRecord]) -> float:
        """Insert multiple records using pipeline and return operation time"""
        start_time = time.time()
        
        pipe = self.client.pipeline()
        for record in records:
            record_dict = asdict(record)
            record_dict['created_at'] = record.created_at.isoformat()
            
            key = f"{self.key_prefix}{record.id}"
            pipe.set(key, json.dumps(record_dict))
        
        pipe.execute()
        return time.time() - start_time
    
    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get a record by ID and return operation time"""
        start_time = time.time()
        
        key = f"{self.key_prefix}{record_id}"
        result = self.client.get(key)
        
        operation_time = time.time() - start_time
        
        if result:
            return json.loads(result), operation_time
        return None, operation_time
    
    def update_record(self, record: KeyValueRecord) -> float:
        """Update a single record and return operation time"""
        # For Redis, update is the same as insert - just overwrite the key
        return self.insert_record(record)
    
    def batch_update_records(self, records: List[KeyValueRecord]) -> float:
        """Update multiple records using pipeline and return operation time"""
        # For Redis, update is the same as batch insert - just overwrite the keys
        return self.batch_insert_records(records)
    
    def clear_data(self):
        """Clear all key-value data with our prefix"""
        keys = self.client.keys(f"{self.key_prefix}*")
        if keys:
            self.client.delete(*keys)

class BenchmarkRunner:
    """Runs performance benchmarks for storage solutions"""
    
    def __init__(self, clickhouse_storage: ClickHouseStorage, redis_storage: RedisStorage):
        self.clickhouse = clickhouse_storage
        self.redis = redis_storage
        self.results: List[BenchmarkResult] = []
    
    def _calculate_stats(self, times: List[float], operation: str, database: str, 
                        records_count: int, success_count: int) -> BenchmarkResult:
        """Calculate performance statistics"""
        total_time = sum(times)
        avg_time = statistics.mean(times) if times else 0
        ops_per_second = records_count / total_time if total_time > 0 else 0
        success_rate = success_count / records_count if records_count > 0 else 0
        
        return BenchmarkResult(
            operation=operation,
            database=database,
            records_count=records_count,
            total_time=total_time,
            avg_time_per_record=avg_time,
            ops_per_second=ops_per_second,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            p95_time=np.percentile(times, 95) if times else 0,
            success_rate=success_rate
        )
    
    def benchmark_single_inserts(self, record_count: int = 1000):
        """Benchmark single record insertions"""
        print(f"Benchmarking single inserts ({record_count} records)...")
        
        # Generate test data
        records = DataGenerator.generate_records(record_count)
        
        # Test ClickHouse
        ch_times = []
        ch_success = 0
        for record in records:
            try:
                time_taken = self.clickhouse.insert_record(record)
                ch_times.append(time_taken)
                ch_success += 1
            except Exception as e:
                print(f"ClickHouse insert error: {e}")
        
        ch_result = self._calculate_stats(ch_times, "single_insert", "clickhouse", 
                                         record_count, ch_success)
        self.results.append(ch_result)
        
        # Test Redis
        redis_times = []
        redis_success = 0
        for record in records:
            try:
                time_taken = self.redis.insert_record(record)
                redis_times.append(time_taken)
                redis_success += 1
            except Exception as e:
                print(f"Redis insert error: {e}")
        
        redis_result = self._calculate_stats(redis_times, "single_insert", "redis", 
                                           record_count, redis_success)
        self.results.append(redis_result)
    
    def benchmark_batch_inserts(self, record_count: int = 1000, batch_size: int = 100):
        """Benchmark batch record insertions"""
        print(f"Benchmarking batch inserts ({record_count} records, batch size {batch_size})...")
        
        # Generate test data
        records = DataGenerator.generate_records(record_count)
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        
        # Test ClickHouse
        ch_times = []
        ch_success = 0
        for batch in batches:
            try:
                time_taken = self.clickhouse.batch_insert_records(batch)
                ch_times.append(time_taken)
                ch_success += len(batch)
            except Exception as e:
                print(f"ClickHouse batch insert error: {e}")
        
        ch_result = self._calculate_stats(ch_times, "batch_insert", "clickhouse", 
                                         record_count, ch_success)
        self.results.append(ch_result)
        
        # Test Redis
        redis_times = []
        redis_success = 0
        for batch in batches:
            try:
                time_taken = self.redis.batch_insert_records(batch)
                redis_times.append(time_taken)
                redis_success += len(batch)
            except Exception as e:
                print(f"Redis batch insert error: {e}")
        
        redis_result = self._calculate_stats(redis_times, "batch_insert", "redis", 
                                           record_count, redis_success)
        self.results.append(redis_result)
    
    def benchmark_reads(self, record_count: int = 1000):
        """Benchmark record reads"""
        print(f"Benchmarking reads ({record_count} records)...")
        
        # First insert some test data
        records = DataGenerator.generate_records(record_count)
        record_ids = [record.id for record in records]
        
        # Insert into both databases
        self.clickhouse.batch_insert_records(records)
        self.redis.batch_insert_records(records)
        
        # Test ClickHouse reads
        ch_times = []
        ch_success = 0
        for record_id in record_ids:
            try:
                result, time_taken = self.clickhouse.get_record(record_id)
                ch_times.append(time_taken)
                if result:
                    ch_success += 1
            except Exception as e:
                print(f"ClickHouse read error: {e}")
        
        ch_result = self._calculate_stats(ch_times, "read", "clickhouse", 
                                         record_count, ch_success)
        self.results.append(ch_result)
        
        # Test Redis reads
        redis_times = []
        redis_success = 0
        for record_id in record_ids:
            try:
                result, time_taken = self.redis.get_record(record_id)
                redis_times.append(time_taken)
                if result:
                    redis_success += 1
            except Exception as e:
                print(f"Redis read error: {e}")
        
        redis_result = self._calculate_stats(redis_times, "read", "redis", 
                                           record_count, redis_success)
        self.results.append(redis_result)
    
    def benchmark_concurrent_operations(self, record_count: int = 500, max_workers: int = 10):
        """Benchmark concurrent read/write operations"""
        print(f"Benchmarking concurrent operations ({record_count} records, {max_workers} workers)...")
        
        records = DataGenerator.generate_records(record_count)
        
        def insert_task_redis(storage, record):
            """Redis insert task - can reuse same connection"""
            try:
                return storage.insert_record(record), True
            except Exception as e:
                print(f"Concurrent Redis insert error: {e}")
                return 0, False
        
        def insert_task_clickhouse(record):
            """ClickHouse insert task - uses thread-local client"""
            try:
                # Use the thread-local client
                if not hasattr(thread_local_data, 'clickhouse_client') or thread_local_data.clickhouse_client is None:
                    print(f"No ClickHouse client available for thread {threading.current_thread().name}")
                    return 0, False
                
                client = thread_local_data.clickhouse_client
                table_name = thread_local_data.table_name
                
                # Perform the insert using the thread-local client
                start_time = time.time()
                data = [[
                    record.id,
                    record.source,
                    record.event_type,
                    record.status,
                    record.string_value_1,
                    record.string_value_2,
                    record.string_value_3,
                    record.string_value_4,
                    record.string_value_5,
                    record.int64_1,
                    record.int64_2,
                    record.int64_3,
                    record.int64_4,
                    record.int64_5,
                    record.dbl_1,
                    record.dbl_2,
                    record.dbl_3,
                    record.dbl_4,
                    record.dbl_5,
                    record.details,
                    record.created_at
                ]]
                
                client.insert(table_name, data)
                operation_time = time.time() - start_time
                
                return operation_time, True
            except Exception as e:
                print(f"Concurrent ClickHouse insert error: {e}")
                return 0, False
        
        # Test concurrent ClickHouse inserts
        print("Testing concurrent ClickHouse inserts...")
        ch_times = []
        ch_success = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            initializer=init_thread_clickhouse_client,
            initargs=(self.clickhouse.host, self.clickhouse.port, self.clickhouse.connection_params, self.clickhouse.table_name)
        ) as executor:
            futures = [executor.submit(insert_task_clickhouse, record) for record in records]
            for future in concurrent.futures.as_completed(futures):
                time_taken, success = future.result()
                ch_times.append(time_taken)
                if success:
                    ch_success += 1
            
            # Clean up thread-local clients when the executor is done
            cleanup_futures = [executor.submit(cleanup_thread_clickhouse_client) for _ in range(max_workers)]
            for future in concurrent.futures.as_completed(cleanup_futures):
                future.result()  # Wait for cleanup to complete
        
        ch_result = self._calculate_stats(ch_times, "concurrent_insert", "clickhouse", 
                                         record_count, ch_success)
        self.results.append(ch_result)
        
        # Test concurrent Redis inserts
        print("Testing concurrent Redis inserts...")
        redis_times = []
        redis_success = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(insert_task_redis, self.redis, record) for record in records]
            for future in concurrent.futures.as_completed(futures):
                time_taken, success = future.result()
                redis_times.append(time_taken)
                if success:
                    redis_success += 1
        
        redis_result = self._calculate_stats(redis_times, "concurrent_insert", "redis", 
                                           record_count, redis_success)
        self.results.append(redis_result)
    
    def benchmark_updates(self, record_count: int = 1000):
        """Benchmark single record updates"""
        print(f"Benchmarking single updates ({record_count} records)...")
        
        # First, insert records to update later
        print("  Inserting initial records...")
        initial_records = DataGenerator.generate_records(record_count)
        
        # Insert initial data in both systems
        for record in initial_records:
            try:
                self.clickhouse.insert_record(record)
                self.redis.insert_record(record)
            except Exception as e:
                print(f"Error during initial insert: {e}")
        
        # Create updated versions of the same records (same IDs, different data)
        updated_records = []
        for record in initial_records:
            # Keep the same ID but generate new data
            updated_record = DataGenerator.generate_record()
            updated_record.id = record.id  # Keep same ID for update
            updated_records.append(updated_record)
        
        # Test ClickHouse updates
        ch_times = []
        ch_success = 0
        print("  Testing ClickHouse updates...")
        for record in updated_records:
            try:
                time_taken = self.clickhouse.update_record(record)
                ch_times.append(time_taken)
                ch_success += 1
            except Exception as e:
                print(f"ClickHouse update error: {e}")
        
        ch_result = self._calculate_stats(ch_times, "single_update", "clickhouse", 
                                         record_count, ch_success)
        self.results.append(ch_result)
        
        # Test Redis updates
        redis_times = []
        redis_success = 0
        print("  Testing Redis updates...")
        for record in updated_records:
            try:
                time_taken = self.redis.update_record(record)
                redis_times.append(time_taken)
                redis_success += 1
            except Exception as e:
                print(f"Redis update error: {e}")
        
        redis_result = self._calculate_stats(redis_times, "single_update", "redis", 
                                           record_count, redis_success)
        self.results.append(redis_result)
    
    def benchmark_batch_updates(self, record_count: int = 5000, batch_size: int = 100):
        """Benchmark batch record updates"""
        print(f"Benchmarking batch updates ({record_count} records, batch size: {batch_size})...")
        
        # First, insert records to update later
        print("  Inserting initial records...")
        initial_records = DataGenerator.generate_records(record_count)
        
        # Insert initial data in both systems using batches
        for i in range(0, len(initial_records), batch_size):
            batch = initial_records[i:i + batch_size]
            try:
                self.clickhouse.batch_insert_records(batch)
                self.redis.batch_insert_records(batch)
            except Exception as e:
                print(f"Error during initial batch insert: {e}")
        
        # Create updated versions of the same records (same IDs, different data)
        updated_records = []
        for record in initial_records:
            # Keep the same ID but generate new data
            updated_record = DataGenerator.generate_record()
            updated_record.id = record.id  # Keep same ID for update
            updated_records.append(updated_record)
        
        # Test ClickHouse batch updates
        ch_times = []
        ch_success = 0
        print("  Testing ClickHouse batch updates...")
        for i in range(0, len(updated_records), batch_size):
            batch = updated_records[i:i + batch_size]
            try:
                time_taken = self.clickhouse.batch_update_records(batch)
                ch_times.append(time_taken)
                ch_success += len(batch)
            except Exception as e:
                print(f"ClickHouse batch update error: {e}")
        
        ch_result = self._calculate_stats(ch_times, "batch_update", "clickhouse", 
                                         record_count, ch_success)
        self.results.append(ch_result)
        
        # Test Redis batch updates
        redis_times = []
        redis_success = 0
        print("  Testing Redis batch updates...")
        for i in range(0, len(updated_records), batch_size):
            batch = updated_records[i:i + batch_size]
            try:
                time_taken = self.redis.batch_update_records(batch)
                redis_times.append(time_taken)
                redis_success += len(batch)
            except Exception as e:
                print(f"Redis batch update error: {e}")
        
        redis_result = self._calculate_stats(redis_times, "batch_update", "redis", 
                                           record_count, redis_success)
        self.results.append(redis_result)
    
    def cleanup(self):
        """Clean up test data"""
        print("Cleaning up test data...")
        try:
            self.clickhouse.clear_table()
            self.redis.clear_data()
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def print_results(self):
        """Print benchmark results in a formatted table"""
        if not self.results:
            print("No benchmark results to display.")
            return
        
        # Convert to DataFrame for better formatting
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        print("\n" + "="*120)
        print("BENCHMARK RESULTS")
        print("="*120)
        
        # Group by operation for cleaner display
        for operation in df['operation'].unique():
            op_data = df[df['operation'] == operation]
            print(f"\n{operation.upper()} OPERATIONS:")
            print("-" * 80)
            
            for _, row in op_data.iterrows():
                print(f"Database: {row['database'].upper()}")
                print(f"  Records: {row['records_count']:,}")
                print(f"  Total Time: {row['total_time']:.3f}s")
                print(f"  Ops/Second: {row['ops_per_second']:,.1f}")
                print(f"  Avg Time/Record: {row['avg_time_per_record']*1000:.3f}ms")
                print(f"  P95 Time: {row['p95_time']*1000:.3f}ms")
                print(f"  Success Rate: {row['success_rate']*100:.1f}%")
                print()

def main():
    """Main function to run the benchmark suite"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark ClickHouse vs Redis key-value performance')
    parser.add_argument('--scenario', choices=['light', 'default', 'heavy'], 
                       default='default', help='Benchmark scenario to run')
    parser.add_argument('--output', default=None, 
                       help='Output file for results (default: auto-generated)')
    args = parser.parse_args()
    
    print("Key-Value Storage Benchmark: ClickHouse vs Redis")
    print("=" * 60)
    print(f"Running scenario: {args.scenario}")
    
    # Get configurations
    config = get_benchmark_config(args.scenario)
    clickhouse_config, redis_config = get_database_configs()
    
    # Initialize storage connections
    try:
        clickhouse = ClickHouseStorage(
            host=clickhouse_config.host,
            port=clickhouse_config.port,
            **clickhouse_config.additional_params
        )
        print("✓ Connected to ClickHouse")
    except Exception as e:
        print(f"✗ Failed to connect to ClickHouse: {e}")
        print("Make sure ClickHouse is running. Run 'python setup_databases.py' to set up.")
        return
    
    try:
        redis_storage = RedisStorage(
            host=redis_config.host,
            port=redis_config.port,
            **redis_config.additional_params
        )
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        print("Make sure Redis is running. Run 'python setup_databases.py' to set up.")
        return
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(clickhouse, redis_storage)
    
    try:
        # Clean any existing data
        if config.cleanup_between_tests:
            runner.cleanup()
        
        print(f"\nRunning benchmark with configuration:")
        print(f"  Single inserts: {config.single_insert_count:,} records")
        print(f"  Batch inserts: {config.batch_insert_count:,} records (batch size: {config.batch_size})")
        print(f"  Single updates: {config.single_insert_count:,} records")
        print(f"  Batch updates: {config.batch_insert_count:,} records (batch size: {config.batch_size})")
        print(f"  Read tests: {config.read_test_count:,} records")
        print(f"  Concurrent tests: {config.concurrent_test_count:,} records ({config.concurrent_workers} workers)")
        print()
        
        # Run different benchmark scenarios with config
        runner.benchmark_single_inserts(record_count=config.single_insert_count)
        
        if config.cleanup_between_tests:
            runner.cleanup()
        
        runner.benchmark_batch_inserts(
            record_count=config.batch_insert_count, 
            batch_size=config.batch_size
        )
        
        runner.benchmark_reads(record_count=config.read_test_count)
        
        if config.cleanup_between_tests:
            runner.cleanup()
        
        # Update benchmarks
        runner.benchmark_updates(record_count=config.single_insert_count)
        
        if config.cleanup_between_tests:
            runner.cleanup()
        
        runner.benchmark_batch_updates(
            record_count=config.batch_insert_count, 
            batch_size=config.batch_size
        )
        
        if config.cleanup_between_tests:
            runner.cleanup()
        
        runner.benchmark_concurrent_operations(
            record_count=config.concurrent_test_count, 
            max_workers=config.concurrent_workers
        )
        
        # Display results
        runner.print_results()
        
        # Save results to file
        if args.output:
            results_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"benchmark_results_{args.scenario}_{timestamp}.json"
        
        if config.save_detailed_results:
            results_data = [asdict(result) for result in runner.results]
            results_data.append({
                'benchmark_config': asdict(config),
                'timestamp': datetime.now().isoformat(),
                'scenario': args.scenario
            })
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"\nDetailed results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if config.cleanup_between_tests:
            runner.cleanup()

if __name__ == "__main__":
    main()
