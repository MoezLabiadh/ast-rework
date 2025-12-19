'''
Script to load AST local datasets into the PG  database.
'''
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import time
import re


# Start timing
start_time = time.time()


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


# Input files configuration
in_loc = r'W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\input_spreadsheets'
in_files = {
    'west_coast': os.path.join(in_loc, 'one_status_west_coast_specific.xlsx'),
    'south_coast': os.path.join(in_loc, 'one_status_south_coast_specific.xlsx'),
    'thompson_okanagan': os.path.join(in_loc, 'one_status_thompson_okanagan_specific.xlsx'),
    'kootenay_boundary': os.path.join(in_loc, 'one_status_kootenay_boundary_specific.xlsx'),
    'cariboo': os.path.join(in_loc, 'one_status_cariboo_specific.xlsx'),
    'skeena': os.path.join(in_loc, 'one_status_skeena_specific.xlsx'),
    'omineca': os.path.join(in_loc, 'one_status_omineca_specific.xlsx'),
    'northeast': os.path.join(in_loc, 'one_status_northeast_specific.xlsx')
}


# Track statistics
loaded_datasets = []
skipped_datasets = []
errors = []


def clean_path(path):
    """Clean path from Excel hyperlink artifacts and extra whitespace"""
    path = path.strip()
    
    # Remove hyperlink brackets and display text (e.g., [text](url) -> url)
    # Pattern: [display_text](actual_url) -> actual_url
    match = re.search(r'\]\((.*?)\)$', path)
    if match:
        path = match.group(1)
    
    # Remove leading/trailing brackets
    path = path.strip('[]')
    
    # Remove extra whitespace and line breaks
    path = ' '.join(path.split())
    
    return path


def convert_to_mapped_drive(path):
    """Convert UNC path to mapped drive"""
    path = path.strip()
    # Replace UNC path with mapped drive (case-insensitive)
    path_lower = path.lower()
    if path_lower.startswith(r'\\spatialfiles.bcgov\work'):
        # Find the position after \work (case-insensitive)
        idx = path_lower.find(r'work') + 4
        path = 'W:' + path[idx:]
    elif path_lower.startswith(r'\\spatialfiles.bcgov/work'):
        # Find the position after /work (case-insensitive)
        idx = path_lower.find(r'work') + 4
        path = 'W:' + path[idx:]
    return path


def get_table_name(datasource):
    """Extract table name from datasource path and truncate to 50 characters (to allow for index names)"""
    path = Path(datasource)
    
    # For shapefiles, remove .shp extension (case-insensitive)
    if path.suffix.lower() == '.shp':
        table_name = path.stem.lower()
    # For GDB feature classes, get the last part (feature class name)
    # Format is usually: path/to/geodatabase.gdb/FeatureClassName
    elif '.gdb' in datasource:
        table_name = datasource.split('.gdb')[-1].strip('/\\').lower()
    else:
        table_name = path.stem.lower()
    
    # Truncate to 50 characters (leaving room for index names like idx_{table}_geometry)
    # PostgreSQL identifier limit is 63, and idx_ + _geometry = 13 characters
    if len(table_name) > 50:
        original_name = table_name
        table_name = table_name[:50]
        print(f"    Note: Table name truncated from {len(original_name)} to 50 characters")
    
    return table_name


def should_skip_dataset(datasource):
    """Check if dataset should be skipped based on name"""
    basename = os.path.basename(datasource).upper()
    
    # Skip if starts with WHSE or REG
    if basename.startswith('WHSE') or basename.startswith('REG'):
        return True
    
    # For GDB feature classes, check the feature class name
    if '.gdb' in datasource:
        fc_name = datasource.split('.gdb')[-1].strip('/\\').upper()
        if fc_name.startswith('WHSE') or fc_name.startswith('REG'):
            return True
    
    return False


def file_exists(datasource):
    """Check if file or GDB exists (handles both shapefiles and featureclasses)"""
    if '.gdb' in datasource:
        # For GDB, only check if the GDB itself exists
        parts = datasource.split('.gdb')
        gdb_path = parts[0] + '.gdb'
        return os.path.exists(gdb_path)
    else:
        # For shapefiles and other formats, check the full path
        return os.path.exists(datasource)


def read_spatial_data(datasource):
    """Read spatial data from shp or GDB featureclass"""
    try:
        # Check for shapefile (case-insensitive)
        if datasource.lower().endswith('.shp'):
            gdf = gpd.read_file(datasource)
        
        elif '.gdb' in datasource:
            # Split path to isolate GDB and featureclass name
            parts = datasource.split('.gdb')
            gdb = parts[0] + '.gdb'
            fc = os.path.basename(datasource)
            gdf = gpd.read_file(filename=gdb, layer=fc)
        
        else:
            raise Exception(f'Format not recognized: {datasource}. Please provide a shp or featureclass (gdb)!')
        
        return gdf
        
    except Exception as e:
        raise Exception(f'Error reading {datasource}: {str(e)}')


def load_spatial_data(datasource, schema, table_name):
    """Load spatial data into PostGIS"""
    try:
        print(f"  Loading {datasource} into {schema}.{table_name}...")
        
        # Read spatial data (handles both shp and GDB featureclasses)
        gdf = read_spatial_data(datasource)
        
        # Ensure geometry column is named 'geometry'
        if gdf.geometry.name != 'geometry':
            gdf = gdf.rename_geometry('geometry')
        
        # Convert to EPSG:3005 (BC Albers) if not already
        if gdf.crs and gdf.crs.to_epsg() != 3005:
            print(f"    Reprojecting from {gdf.crs.to_epsg()} to EPSG:3005...")
            gdf = gdf.to_crs(epsg=3005)
        
        # Load to PostGIS
        gdf.to_postgis(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists='replace',
            index=False
        )
        
        # Create spatial index
        with engine.connect() as conn:
            index_name = f"sidx_{table_name}_geom"[:63]  # Truncate index name if needed
            sql = text(f'CREATE INDEX IF NOT EXISTS {index_name} ON {schema}.{table_name} USING GIST (geometry);')
            conn.execute(sql)
            conn.commit()
        
        print(f"    ✓ Successfully loaded {len(gdf)} features with spatial index")
        return True
        
    except Exception as e:
        print(f"    ✗ Error loading dataset: {str(e)}")
        return False


# Process each region
for schema, excel_file in in_files.items():
    print(f"\n{'='*60}")
    print(f"Processing {schema.upper()} region: {excel_file}")
    print('='*60)
    
    # Track datasources within this spreadsheet only
    processed_in_spreadsheet = set()
    
    # Read the Excel file
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        errors.append(f"{schema}: Could not read Excel file - {e}")
        continue
    
    # Check if Datasource column exists
    if 'Datasource' not in df.columns:
        print(f"Warning: 'Datasource' column not found in {excel_file}")
        errors.append(f"{schema}: 'Datasource' column not found")
        continue
    
    # Process each datasource
    for datasource in df['Datasource'].dropna():
        # Strip, clean, and convert datasource
        datasource = str(datasource).strip()
        datasource = clean_path(datasource)
        datasource = convert_to_mapped_drive(datasource)
        
        # Skip if starts with WHSE or REG (don't print or track these)
        if should_skip_dataset(datasource):
            continue
        
        # Check if already processed in THIS spreadsheet only
        if datasource in processed_in_spreadsheet:
            print(f"Skipping {datasource} (duplicate in same spreadsheet)")
            skipped_datasets.append(f"{schema}: {datasource} (duplicate in spreadsheet)")
            continue
        
        # Check if file/GDB exists
        if not file_exists(datasource):
            print(f"Skipping {datasource} (file not found)")
            errors.append(f"{schema}: {datasource} (file not found)")
            continue
        
        # Get table name (with truncation to 63 chars)
        table_name = get_table_name(datasource)
        
        # Load the data
        print(f"Processing: {os.path.basename(datasource)}")
        success = load_spatial_data(datasource, schema, table_name)
        
        if success:
            processed_in_spreadsheet.add(datasource)
            loaded_datasets.append(f"{schema}.{table_name}")
        else:
            errors.append(f"{schema}.{table_name}: Failed to load {datasource}")


# Calculate processing time
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)


# Print summary
print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"\nSuccessfully loaded: {len(loaded_datasets)} datasets")
for dataset in loaded_datasets:
    print(f"  ✓ {dataset}")


print(f"\nSkipped: {len(skipped_datasets)} datasets")
for dataset in skipped_datasets:
    print(f"  - {dataset}")


if errors:
    print(f"\nErrors: {len(errors)}")
    for error in errors:
        print(f"  ✗ {error}")


print("\n" + "="*60)
print(f"Processing Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
print("="*60)
print("Loading complete!")