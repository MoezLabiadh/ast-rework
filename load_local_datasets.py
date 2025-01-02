"""
Script to load/update the AST local regional datasets into a database (duckdb)
"""

import warnings
warnings.simplefilter(action='ignore')

import os
import timeit
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt


class DuckDBConnector:
    def __init__(self, db=':memory:'):
        self.db = db
        self.conn = None
    
    def connect_to_db(self):
        """Connects to a DuckDB database and installs spatial extension."""
        self.conn = duckdb.connect(self.db)
        self.conn.install_extension('spatial')
        self.conn.load_extension('spatial')
        return self.conn
    
    def disconnect_db(self):
        """Disconnects from the DuckDB database."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            

def create_db_schemas (conn, schema_list):
    """
    Creates schemas for each region in the DB
    """
    for schema in schema_list:
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS {schema};')
    
      
    
def esri_to_gdf(file_path):
    """Returns a Geopandas file (gdf) from
       an ESRI format vector (shp or featureclass/gdb)
    """
    
    if '.shp' in file_path.lower():
        gdf = gpd.read_file(file_path)
    elif '.gdb' in file_path:
        l = file_path.split('.gdb')
        gdb = l[0] + '.gdb'
        fc = os.path.basename(file_path)
        gdf = gpd.read_file(filename=gdb, layer=fc)
    else:
        raise Exception('Format not recognized. Please provide a shp or featureclass (gdb)!')
        
    return gdf


def format_geom_col(gdf):
    """
    Transform the geometry column of a geodataframe to WKT
    
    """
    gdf['GEOMETRY']= gdf['geometry'].apply(lambda x: wkt.dumps(x, output_dimension=2))
    gdf['GEOMETRY'] = gdf['GEOMETRY'].astype(str)
    
    gdf = gdf.drop(columns=['geometry'])
    
    return gdf



def add_data_to_duckdb(conn, gdf, schema, table):
    
    conn.execute(f"SET search_path TO {schema};")
    
    dck_tab_list = conn.execute(
        """SELECT* FROM information_schema.tables;"""
    ).df()['table_name'].to_list()

    if table in dck_tab_list:
        dck_row_count = conn.execute(
            f"""SELECT COUNT(*) FROM {schema}.{table}"""
        ).fetchone()[0]
        
        dck_col_nams = conn.execute(
            f"""SELECT* FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE table_schema = '{schema}' AND table_name = '{table}'"""
            ).df()['column_name'].to_list()

        if (dck_row_count != len(gdf)) or (set(list(gdf.columns)) != set(dck_col_nams)):
            print(f'....import to Duckdb ({gdf.shape[0]} rows)')
            chunk_size = 10000
            total_chunks = (len(gdf) + chunk_size - 1) // chunk_size
            
            # Process the initial chunk and create the table
            initial_chunk = gdf.iloc[0:chunk_size]
            print(f'.......processing chunk 1 of {total_chunks}')
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table} AS
                SELECT * EXCLUDE GEOMETRY, ST_GeomFromText(GEOMETRY) AS GEOMETRY
                FROM initial_chunk;
            """
            conn.execute(create_table_query)
            
            # Process and append the rest of the chunks
            for i in range(chunk_size, len(gdf), chunk_size):
                chunk = gdf.iloc[i:i + chunk_size]
                chunk_number = (i // chunk_size) + 1
                print(f'.......processing chunk {chunk_number} of {total_chunks}')
                
                insert_query = f"""
                INSERT INTO {table}
                    SELECT * EXCLUDE GEOMETRY, ST_GeomFromText(GEOMETRY) AS GEOMETRY
                    FROM chunk;
                """
                conn.execute(insert_query)
                
            #Add a spatial index to the table
            print('.......creating a spatial RTREE index')
            conn.execute(
                f"""DROP INDEX IF EXISTS idx_{schema}_{table};
                    CREATE INDEX idx_{schema}_{table} ON {schema}.{table} USING RTREE (GEOMETRY);""")
        else:
            print('....data already in db: skip importing')
            pass

    else:
        print(f'....import to Duckdb ({gdf.shape[0]} rows)')
        chunk_size = 10000
        total_chunks = (len(gdf) + chunk_size - 1) // chunk_size
        
        # Process the initial chunk and create the table
        initial_chunk = gdf.iloc[0:chunk_size]
        print(f'.......processing chunk 1 of {total_chunks}')
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table} AS
            SELECT * EXCLUDE GEOMETRY, ST_GeomFromText(GEOMETRY) AS GEOMETRY
            FROM initial_chunk;
        """
        conn.execute(create_table_query)
        
        # Process and append the rest of the chunks
        for i in range(chunk_size, len(gdf), chunk_size):
            chunk = gdf.iloc[i:i + chunk_size]
            chunk_number = (i // chunk_size) + 1
            print(f'.......processing chunk {chunk_number} of {total_chunks}')
            
            insert_query = f"""
            INSERT INTO {table}
                SELECT * EXCLUDE GEOMETRY, ST_GeomFromText(GEOMETRY) AS GEOMETRY
                FROM chunk;
            """
            conn.execute(insert_query)
            
        #Add a spatial index to the table
        print('.......creating a spatial RTREE index')
        conn.execute(
            f"""DROP INDEX IF EXISTS idx_{schema}_{table};
                CREATE INDEX idx_{schema}_{table} ON {schema}.{table} USING RTREE (GEOMETRY);""")
    
   
    
def load_datasets(in_files, conn):
    
    total = len(in_files)
    count_reg= 1
    
    for k, v in in_files.items():
        print (f'\n\nLoading data for region {count_reg} of {total}: {k}')
        
        schema= k.lower()
        
        df = pd.read_excel(v)
        df.rename(columns={
            "Featureclass_Name(valid characters only)": "Dataset"},
            inplace=True)
    
        count = len(df)
    
        for i, row in df.iterrows():
            print(f"\n..reading dataset {i+1} of {count}: {row['Dataset']}")
            dataset= row['Dataset']
            datasource = row['Datasource']
            datasource = datasource.strip() if isinstance(datasource, str) else datasource
    
            # Skip datasources that are empty and BCGW datasets
            if pd.isna(datasource) or datasource.startswith("WHSE") or datasource.startswith("REG"):
                continue
    
            # Read the data
            try:
                gdf = esri_to_gdf(datasource)
                gdf = format_geom_col(gdf)
                table_name = dataset.lower().replace(" ", "_")
            
            except:
                print("datasource is Invalid. Skipping!")
            
            # Load the data
            if len(gdf) > 0:
                add_data_to_duckdb(conn, gdf, schema, table_name)
            else:
                print ('....No data to add.')
        
        count_reg += 1
    






if __name__ == "__main__":
    start_t = timeit.default_timer()
    
    wks= r'W:\lwbc\visr\Workarea\moez_labiadh\STATUSING\ast_rework\local_datasets\database'
    print ('Connecting to Duckdb') 
    projDB= os.path.join(wks,  'ast_local_datasets.db')
    Duckdb= DuckDBConnector(db= projDB)
    Duckdb.connect_to_db()
    conn= Duckdb.conn 
    conn.execute("SET GLOBAL pandas_analyze_sample=1000000")


    # input AST spreadsheets
    in_loc = r'P:\corp\script_whse\python\Utility_Misc\Ready\statusing_tools_arcpro\statusing_input_spreadsheets'
    in_files= {
        'RWC': os.path.join(in_loc, 'one_status_west_coast_specific.xlsx'),
        'RSC': os.path.join(in_loc, 'one_status_south_coast_specific.xlsx'),
        'RTO': os.path.join(in_loc, 'one_status_thompson_okanagan_specific.xlsx'),
        'RKB': os.path.join(in_loc, 'one_status_kootenay_boundary_specific.xlsx'),
        'RCB': os.path.join(in_loc, 'one_status_cariboo_specific.xlsx'),
        'RSK': os.path.join(in_loc, 'one_status_skeena_specific.xlsx'),
        'ROM': os.path.join(in_loc, 'one_status_omineca_specific.xlsx'),
        'RNO': os.path.join(in_loc, 'one_status_northeast_specific.xlsx')   
    }
    
    try:
        print ('\nReviewing DB schemas')
        schema_list= [schema.lower() for schema in in_files.keys()]
        create_db_schemas (conn, schema_list)
        
        print ('\nLoading datasets')
        load_datasets(in_files, conn)
        
        
    except Exception as e:
        raise Exception(f"Error occurred: {e}")  

    finally: 
        Duckdb.disconnect_db()
        
        finish_t = timeit.default_timer()
        t_sec = round(finish_t - start_t)
        mins, secs = divmod(t_sec, 60)
        print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')

