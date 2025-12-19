'''
Script to create a PG database with PostGIS extension and multiple schemas.
Each schema will hold data for a specific region.
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
schemas = ['west_coast', 'south_coast', 'thompson_okanagan', 'kootenay_boundary', 'cariboo', 'skeena', 'omineca', 'northeast']

for schema in schemas:
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    print(f"Schema '{schema}' created!")

conn.commit()

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

cur.close()
conn.close()

print("\nDatabase setup complete!")