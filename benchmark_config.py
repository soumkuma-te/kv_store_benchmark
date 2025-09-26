#!/usr/bin/env python3
"""
Configuration settings for the key-value benchmark suite.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    additional_params: Dict[str, Any]

@dataclass
class BenchmarkConfig:
    """Benchmark test configuration"""
    # Test data sizes
    single_insert_count: int = 1000
    batch_insert_count: int = 5000
    batch_size: int = 100
    read_test_count: int = 1000
    concurrent_test_count: int = 500
    concurrent_workers: int = 10
    
    # Test execution settings
    warmup_iterations: int = 3
    cleanup_between_tests: bool = True
    save_detailed_results: bool = True
    
    # Data generation settings
    string_length_range: tuple = (10, 100)
    list_size_range: tuple = (3, 10)
    int_value_range: tuple = (1, 1000000)
    float_value_range: tuple = (0.0, 1000.0)

# Database configurations
CLICKHOUSE_CONFIG = DatabaseConfig(
    host='localhost',
    port=8123,
    additional_params={
        'database': 'benchmark_db',
        'username': 'benchmark_user',
        'password': 'benchmark_password',
        'connect_timeout': 30,
        'send_receive_timeout': 300
    }
)

REDIS_CONFIG = DatabaseConfig(
    host='localhost',
    port=6379,
    additional_params={
        'db': 0,
        'decode_responses': True,
        'socket_timeout': 30,
        'socket_connect_timeout': 30,
        'retry_on_timeout': True
    }
)

# Default benchmark configuration
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()

# Performance test configurations for different scenarios
LIGHT_BENCHMARK_CONFIG = BenchmarkConfig(
    single_insert_count=500,
    batch_insert_count=2000,
    batch_size=50,
    read_test_count=500,
    concurrent_test_count=250,
    concurrent_workers=5
)

HEAVY_BENCHMARK_CONFIG = BenchmarkConfig(
    single_insert_count=5000,
    batch_insert_count=50000,
    batch_size=500,
    read_test_count=5000,
    concurrent_test_count=2000,
    concurrent_workers=20
)

# Test scenarios mapping
BENCHMARK_SCENARIOS = {
    'light': LIGHT_BENCHMARK_CONFIG,
    'default': DEFAULT_BENCHMARK_CONFIG,
    'heavy': HEAVY_BENCHMARK_CONFIG
}

def get_benchmark_config(scenario: str = 'default') -> BenchmarkConfig:
    """Get benchmark configuration for a specific scenario"""
    return BENCHMARK_SCENARIOS.get(scenario, DEFAULT_BENCHMARK_CONFIG)

def get_database_configs() -> tuple:
    """Get database configurations"""
    return CLICKHOUSE_CONFIG, REDIS_CONFIG
