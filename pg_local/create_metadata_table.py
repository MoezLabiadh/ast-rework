'''
Script to initialize the AST local datasets database.

Creates the metadata table for tracking dataset load times and source file info.
Run this script once before running the main loader script.

Author: Moez Labiadh, GeoBC

Date: December 2025

'''
import os
from sqlalchemy import create_engine, text


# Database connection parameters
db_params = {
    'host': 'localhost',
    'database': 'ast_local_datasets',
    'user': 'postgres',
    'password': os.getenv('PG_LCL_SUSR_PASS')
}


# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['database']}"
)


def create_metadata_table():
    """Create metadata table to track dataset load times and source file info"""
    create_sql = text("""
        CREATE TABLE IF NOT EXISTS public.dataset_metadata (
            schema_name VARCHAR(100),
            table_name VARCHAR(100),
            datasource VARCHAR(1000),
            last_modified TIMESTAMP,
            last_loaded TIMESTAMP,
            feature_count INTEGER,
            PRIMARY KEY (schema_name, table_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_metadata_schema 
            ON public.dataset_metadata(schema_name);
        
        CREATE INDEX IF NOT EXISTS idx_metadata_last_modified 
            ON public.dataset_metadata(last_modified);
    """)
    
    with engine.connect() as conn:
        conn.execute(create_sql)
        conn.commit()
    
    print("✓ Metadata table created successfully")
    print("  - Table: public.dataset_metadata")
    print("  - Indexes: schema_name, last_modified")


if __name__ == "__main__":
    print("="*60)
    print("AST Database Initialization")
    print("="*60)
    
    print("\nCreating metadata table...")
    create_metadata_table()
    
    print("\n" + "="*60)
    print("Database initialization complete!")
    print("="*60)
    print("\nYou can now run the main loader script.")