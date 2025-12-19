import os
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from pathlib import Path
import time

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
    'rwc': os.path.join(in_loc, 'one_status_west_coast_specific.xlsx'),
    'rsc': os.path.join(in_loc, 'one_status_south_coast_specific.xlsx'),
    'rto': os.path.join(in_loc, 'one_status_thompson_okanagan_specific.xlsx'),
    'rkb': os.path.join(in_loc, 'one_status_kootenay_boundary_specific.xlsx'),
    'rcb': os.path.join(in_loc, 'one_status_cariboo_specific.xlsx'),
    'rsk': os.path.join(in_loc, 'one_status_skeena_specific.xlsx'),
    'rom': os.path.join(in_loc, 'one_status_omineca_specific.xlsx'),
    'rno': os.path.join(in_loc, 'one_status_northeast_specific.xlsx')
}

# Track statistics
loaded_datasets = []
skipped_datasets = []
errors = []

def get_table_name(datasource):
    """Extract table name from datasource path"""
    path = Path(datasource)
    
    # For shapefiles, remove .shp extension
    if path.suffix.lower() == '.shp':
        return path.stem.lower()
    
    # For GDB feature classes, get the last part (feature class name)
    # Format is usually: path/to/geodatabase.gdb/FeatureClassName
    if '.gdb' in datasource:
        return datasource.split('.gdb')[-1].strip('/\\').lower()
    
    return path.stem.lower()

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

def load_spatial_data(datasource, schema, table_name):
    """Load spatial data into PostGIS"""
    try:
        print(f"  Loading {datasource} into {schema}.{table_name}...")
        
        # Read spatial data
        gdf = gpd.read_file(datasource)
        
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
        
        print(f"    ✓ Successfully loaded {len(gdf)} features")
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
    for idx, datasource in enumerate(df['Datasource'].dropna(), 1):
        datasource = str(datasource).strip()
        
        # Skip if starts with WHSE or REG
        if should_skip_dataset(datasource):
            print(f"{idx}. Skipping {datasource} (starts with WHSE or REG)")
            skipped_datasets.append(f"{schema}: {datasource} (WHSE/REG)")
            continue
        
        # Check if already processed in THIS spreadsheet only
        if datasource in processed_in_spreadsheet:
            print(f"{idx}. Skipping {datasource} (duplicate in same spreadsheet)")
            skipped_datasets.append(f"{schema}: {datasource} (duplicate in spreadsheet)")
            continue
        
        # Check if file exists
        if not os.path.exists(datasource):
            print(f"{idx}. Skipping {datasource} (file not found)")
            errors.append(f"{schema}: {datasource} (file not found)")
            continue
        
        # Get table name
        table_name = get_table_name(datasource)
        
        # Load the data
        print(f"{idx}. Processing: {os.path.basename(datasource)}")
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
print(f"Total Time: {total_time:.2f} seconds")
print("="*60)
print("Loading complete!")