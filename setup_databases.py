#!/usr/bin/env python3
"""
Database Setup Script for Key-Value Benchmark
Sets up ClickHouse and Redis databases for benchmarking.
"""

import sys
import time
import subprocess
from typing import Optional

import clickhouse_connect
import redis

def check_clickhouse_connection(host: str = 'localhost', port: int = 8123) -> bool:
    """Check if ClickHouse is accessible"""
    try:
        print(f"Checking ClickHouse connection to {host}:{port}")
        client = clickhouse_connect.get_client(
            host=host, 
            port=port,
            username='benchmark_user',
            password='benchmark_password',
            database='benchmark_db'
        )
        client.ping()
        return True
    except Exception as e:
        print(f"ClickHouse connection failed: {e}")
        return False

def check_redis_connection(host: str = 'localhost', port: int = 6379) -> bool:
    """Check if Redis is accessible"""
    try:
        print(f"Checking Redis connection to {host}:{port}")
        client = redis.Redis(host=host, port=port, decode_responses=True)
        client.ping()
        return True
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return False

def start_clickhouse_docker() -> bool:
    """Start ClickHouse using Docker"""
    try:
        print("Starting ClickHouse with Docker...")
        cmd = [
            "docker", "run", "-d", "--name", "clickhouse-benchmark",
            "-p", "8123:8123", "-p", "9000:9000",
            "clickhouse/clickhouse-server"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Wait for ClickHouse to be ready
        print("Waiting for ClickHouse to start...")
        for i in range(30):
            if check_clickhouse_connection():
                print("✓ ClickHouse is ready")
                return True
            time.sleep(2)
        
        print("✗ ClickHouse failed to start within timeout")
        return False
        
    except subprocess.CalledProcessError as e:
        if b"already in use" in e.stderr:
            print("ClickHouse container already exists, checking if it's running...")
            try:
                subprocess.run(["docker", "start", "clickhouse-benchmark"], 
                             check=True, capture_output=True)
                time.sleep(5)
                return check_clickhouse_connection()
            except:
                pass
        print(f"Failed to start ClickHouse: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error starting ClickHouse: {e}")
        return False

def start_redis_docker() -> bool:
    """Start Redis using Docker"""
    try:
        print("Starting Redis with Docker...")
        cmd = [
            "docker", "run", "-d", "--name", "redis-benchmark",
            "-p", "6379:6379",
            "redis:latest"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Wait for Redis to be ready
        print("Waiting for Redis to start...")
        for i in range(15):
            if check_redis_connection():
                print("✓ Redis is ready")
                return True
            time.sleep(1)
        
        print("✗ Redis failed to start within timeout")
        return False
        
    except subprocess.CalledProcessError as e:
        if b"already in use" in e.stderr:
            print("Redis container already exists, checking if it's running...")
            try:
                subprocess.run(["docker", "start", "redis-benchmark"], 
                             check=True, capture_output=True)
                time.sleep(2)
                return check_redis_connection()
            except:
                pass
        print(f"Failed to start Redis: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error starting Redis: {e}")
        return False

def setup_clickhouse_schema():
    """Set up initial ClickHouse schema"""
    try:
        # Connect with credentials to create database
        client = clickhouse_connect.get_client(
            username='benchmark_user',
            password='benchmark_password'
        )
        
        # Create database if it doesn't exist
        client.command("CREATE DATABASE IF NOT EXISTS benchmark_db")
        
        # Use the benchmark database with full credentials
        client = clickhouse_connect.get_client(
            username='benchmark_user',
            password='benchmark_password',
            database='benchmark_db'
        )
        
        # Create the key_value_store table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS key_value_store (
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
        """
        
        client.command(create_table_sql)
        print("✓ ClickHouse schema created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create ClickHouse schema: {e}")
        return False

def check_docker_availability() -> bool:
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], 
                      check=True, capture_output=True)
        return True
    except:
        return False

def main():
    """Main setup function"""
    print("Key-Value Benchmark Database Setup")
    print("=" * 40)
    
    # Check if Docker is available
    if not check_docker_availability():
        print("✗ Docker is not available. Please install Docker or set up databases manually.")
        sys.exit(1)
    
    # Check current database status
    clickhouse_ready = check_clickhouse_connection()
    redis_ready = check_redis_connection()
    
    if clickhouse_ready and redis_ready:
        print("✓ Both databases are already running")
    else:
        print("Setting up databases...")
        
        # Start ClickHouse if needed
        if not clickhouse_ready:
            if not start_clickhouse_docker():
                print("✗ Failed to start ClickHouse")
                sys.exit(1)
        else:
            print("✓ ClickHouse is already running")
        
        # Start Redis if needed
        if not redis_ready:
            if not start_redis_docker():
                print("✗ Failed to start Redis")
                sys.exit(1)
        else:
            print("✓ Redis is already running")
    
    # Set up ClickHouse schema
    if not setup_clickhouse_schema():
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("✓ Database setup completed successfully!")
    print("\nYou can now run the benchmark:")
    print("python key_value_benchmark.py")
    print("\nTo stop the databases later:")
    print("docker stop clickhouse-benchmark redis-benchmark")
    print("docker rm clickhouse-benchmark redis-benchmark")

if __name__ == "__main__":
    main()
