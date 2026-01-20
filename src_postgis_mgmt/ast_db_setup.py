'''
Script to create a PG database with PostGIS extension and multiple schemas.

Each schema will hold data for a specific region.

Also creates the metadata table for tracking dataset load times.

Author: Moez Labiadh, GeoBC

Created: 2025-12-20
'''
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Connect to PG
conn = psycopg2.connect(
    host="localhost",
    user="postgres",  
    password=os.getenv('PG_LCL_SUSR_PASS'),  
    database="postgres"  
)

# Set autocommit to allow database creation
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

# Create a cursor
cur = conn.cursor()

# Create the database
db_name = "ast_local_datasets"
cur.execute(f"CREATE DATABASE {db_name};")

print(f"Database '{db_name}' created successfully!")

# Close connection to postgres database
cur.close()
conn.close()

# Now connect to the new database and enable PostGIS
conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password=os.getenv('PG_LCL_SUSR_PASS'),
    database=db_name
)

cur = conn.cursor()

# Enable PostGIS extension
cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
conn.commit()

print("PostGIS extension enabled!")

# Verify PostGIS is installed
cur.execute("SELECT PostGIS_version();")
version = cur.fetchone()
print(f"PostGIS version: {version[0]}")

# Create schemas for each region
schemas = [
    'west_coast', 'south_coast', 'thompson_okanagan', 
    'kootenay_boundary', 'cariboo', 'skeena', 'omineca', 'northeast'
]

for schema in schemas:
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    print(f"Schema '{schema}' created!")

conn.commit()

# Create metadata table for tracking dataset loads
print("\nCreating metadata table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS public.dataset_metadata (
        schema_name VARCHAR(100),
        table_name VARCHAR(100),
        datasource VARCHAR(1000),
        last_modified TIMESTAMP,
        last_loaded TIMESTAMP,
        feature_count INTEGER,
        PRIMARY KEY (schema_name, table_name)
    );
""")

# Create indexes for better query performance
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_metadata_schema 
        ON public.dataset_metadata(schema_name);
""")

cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_metadata_last_modified 
        ON public.dataset_metadata(last_modified);
""")

conn.commit()

print("✓ Metadata table created successfully")
print("  - Table: public.dataset_metadata")
print("  - Indexes: schema_name, last_modified")

# List all schemas to verify
cur.execute("""
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    ORDER BY schema_name;
""")
all_schemas = cur.fetchall()
print("\nAll schemas in database:")
for schema in all_schemas:
    print(f"  - {schema[0]}")

# List all tables in public schema
cur.execute("""
    SELECT tablename 
    FROM pg_tables 
    WHERE schemaname = 'public'
    ORDER BY tablename;
""")
all_tables = cur.fetchall()
print("\nTables in public schema:")
for table in all_tables:
    print(f"  - {table[0]}")

cur.close()
conn.close()

print("\nDatabase setup complete!")
print("You can now run the AST dataset loader script.")