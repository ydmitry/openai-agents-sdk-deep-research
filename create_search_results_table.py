#!/usr/bin/env python3
"""
Migration script to create search_results table for storing search results with embeddings.

This script creates the necessary database structure for the postgres collector
in the clarification agent system.

Usage:
    python create_search_results_table.py
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import sys
from datetime import datetime

# Database configuration (same as demo_pgvector.py)
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 5433,
    'database': 'my-deep-research',
    'user': 'postgres',
    'password': 'secret'
}

def connect_to_db():
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def create_search_results_table(conn):
    """Create the search_results table with pgvector support."""
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            print("🔄 Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create search_results table
            print("🔄 Creating search_results table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    id SERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    text TEXT NOT NULL,
                    embedding vector(1536)
                );
            """)
            
            # Create indexes for better performance
            print("🔄 Creating indexes...")
            
            # Index on session_id for session-based queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS search_results_session_id_idx 
                ON search_results (session_id);
            """)
            
            # Index on created_at for time-based queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS search_results_created_at_idx 
                ON search_results (created_at);
            """)
            
            # Index on embedding column for similarity searches (only if embedding is not null)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS search_results_embedding_idx 
                ON search_results USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                WHERE embedding IS NOT NULL;
            """)
            
            conn.commit()
            print("✅ Table 'search_results' created successfully")
            print("✅ All indexes created successfully")
            
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        conn.rollback()
        raise

def verify_table_structure(conn):
    """Verify the table was created correctly."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check table exists and get column info
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'search_results'
                ORDER BY ordinal_position;
            """)
            
            columns = cur.fetchall()
            
            if not columns:
                print("❌ Table 'search_results' not found")
                return False
            
            print("\n📊 Table structure verification:")
            print("-" * 50)
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  {col['column_name']:<15} {col['data_type']:<20} {nullable}")
            
            # Check indexes
            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes 
                WHERE tablename = 'search_results'
                ORDER BY indexname;
            """)
            
            indexes = cur.fetchall()
            
            if indexes:
                print("\n🔍 Indexes:")
                print("-" * 50)
                for idx in indexes:
                    print(f"  {idx['indexname']}")
            
            print("-" * 50)
            print("✅ Table structure verified successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error verifying table structure: {e}")
        return False

def main():
    """Main function to run the migration."""
    print("🚀 Starting search_results table migration...")
    print(f"📅 Migration timestamp: {datetime.now().isoformat()}")
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        print("❌ Failed to connect to database")
        sys.exit(1)
    
    print(f"✅ Connected to PostgreSQL database: {DB_CONFIG['database']}")
    
    try:
        # Create table and indexes
        create_search_results_table(conn)
        
        # Verify the table was created correctly
        if verify_table_structure(conn):
            print("\n🎉 Migration completed successfully!")
            print("   The search_results table is ready for use with the postgres collector.")
        else:
            print("\n❌ Migration verification failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()
        print("\n👋 Database connection closed")

if __name__ == "__main__":
    main() 