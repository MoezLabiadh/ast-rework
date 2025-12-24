"""
Script to load AST local datasets into the PG database.

Reads regional Excel spreadsheets, loads spatial datasets (shapefiles/GDB feature classes)
into schema-separated tables, reprojects to BC Albers (EPSG:3005), creates spatial
indexes, and handles path conversion, name sanitization, and duplicate detection.

Only reloads datasets if the source file has been modified since last load:
    This script overwrites datasets based **ONLY** on latest file modification **DATE** (not time).

Author: Moez Labiadh, GeoBC

Date: December 2025
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from datetime import datetime, date
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
unchanged_datasets = []
errors = []


def get_file_modified_date(datasource):
    """Get the last modified DATE (not time) of a file or GDB"""
    try:
        if '.gdb' in datasource:
            # For GDB, check the GDB directory's modification date
            parts = datasource.split('.gdb')
            gdb_path = parts[0] + '.gdb'
            mod_datetime = datetime.fromtimestamp(os.path.getmtime(gdb_path))
            return mod_datetime.date()

        elif datasource.lower().endswith('.shp'):
            # For shapefiles, check all component files and return the most recent date
            base_path = datasource[:-4]  # Remove .shp extension
            shapefile_extensions = ['.shp', '.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']

            mod_dates = []
            for ext in shapefile_extensions:
                component_file = base_path + ext
                if os.path.exists(component_file):
                    mod_datetime = datetime.fromtimestamp(os.path.getmtime(component_file))
                    mod_dates.append(mod_datetime.date())

            if mod_dates:
                # Return the most recent modification date among all components
                return max(mod_dates)
            else:
                # Fallback to just the .shp file
                mod_datetime = datetime.fromtimestamp(os.path.getmtime(datasource))
                return mod_datetime.date()

        else:
            # For other formats
            mod_datetime = datetime.fromtimestamp(os.path.getmtime(datasource))
            return mod_datetime.date()

    except Exception as e:
        print(f"    Warning: Could not get modified date for {datasource}: {e}")
        return None


def get_dataset_metadata(schema, table_name):
    """Retrieve metadata for a specific dataset"""
    query = text("""
        SELECT datasource, last_modified, last_loaded, feature_count
        FROM public.dataset_metadata
        WHERE schema_name = :schema AND table_name = :table_name
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"schema": schema, "table_name": table_name})
        row = result.fetchone()
        if row:
            return {
                'datasource': row[0],
                'last_modified': row[1],
                'last_loaded': row[2],
                'feature_count': row[3]
            }
    return None


def update_dataset_metadata(schema, table_name, datasource, last_modified, feature_count):
    """Update metadata for a dataset"""
    query = text("""
        INSERT INTO public.dataset_metadata 
            (schema_name, table_name, datasource, last_modified, last_loaded, feature_count)
        VALUES 
            (:schema, :table_name, :datasource, :last_modified, :last_loaded, :feature_count)
        ON CONFLICT (schema_name, table_name) 
        DO UPDATE SET
            datasource = EXCLUDED.datasource,
            last_modified = EXCLUDED.last_modified,
            last_loaded = EXCLUDED.last_loaded,
            feature_count = EXCLUDED.feature_count;
    """)

    with engine.connect() as conn:
        conn.execute(query, {
            "schema": schema,
            "table_name": table_name,
            "datasource": datasource,
            "last_modified": last_modified,
            "last_loaded": datetime.now(),
            "feature_count": feature_count
        })
        conn.commit()


def needs_update(schema, table_name, datasource, current_modified_date):
    """Check if dataset needs to be updated based on file modification DATE only (ignores time)"""
    metadata = get_dataset_metadata(schema, table_name)

    # If no metadata exists, needs update
    if metadata is None:
        print(f"    → New dataset, will load")
        return True

    # If datasource path changed, needs update
    if metadata['datasource'] != datasource:
        print(f"    → Datasource path changed, will reload")
        return True

    # If we can't get modified date, assume needs update
    if current_modified_date is None:
        print(f"    → Cannot determine modified date, will reload")
        return True

    # Compare dates only (extract date from datetime if needed)
    stored_modified = metadata['last_modified']
    if stored_modified is not None:
        if isinstance(stored_modified, datetime):
            stored_modified_date = stored_modified.date()
        else:
            stored_modified_date = stored_modified
    else:
        stored_modified_date = None

    # If file was modified on a different date, needs update
    if stored_modified_date is None or current_modified_date > stored_modified_date:
        print(f"    → Source file modified, will reload")
        print(f"       File modified date: {current_modified_date}")
        print(f"       Previous load date: {stored_modified_date}")
        return True

    # Otherwise, skip update
    print(f"    → No changes detected since last load ({metadata['last_loaded']}), skipping")
    return False


def clean_path(path):
    """Clean path from Excel hyperlink artifacts and extra whitespace"""
    path = path.strip()

    # Remove hyperlink brackets and display text (e.g., [text](url) -> url)
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
        idx = path_lower.find(r'work') + 4
        path = 'W:' + path[idx:]
    elif path_lower.startswith(r'\\spatialfiles.bcgov/work'):
        idx = path_lower.find(r'work') + 4
        path = 'W:' + path[idx:]
    return path


def get_table_name(datasource):
    """Extract table name from datasource path and truncate to 50 characters"""
    path = Path(datasource)

    if path.suffix.lower() == '.shp':
        table_name = path.stem
    elif '.gdb' in datasource:
        after_gdb = datasource.split('.gdb')[-1]
        after_gdb = after_gdb.strip('/\\')
        table_name = after_gdb.split('\\')[-1].split('/')[-1]
    else:
        table_name = path.stem

    table_name = table_name.lower()
    table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
    table_name = re.sub(r'_+', '_', table_name)
    table_name = table_name.strip('_')

    if table_name and table_name[0].isdigit():
        table_name = 't_' + table_name

    if len(table_name) > 50:
        original_length = len(table_name)
        table_name = table_name[:50].rstrip('_')
        print(f"    Note: Table name truncated from {original_length} to {len(table_name)} characters")

    return table_name


def should_skip_dataset(datasource):
    """Check if dataset should be skipped based on name"""
    basename = os.path.basename(datasource).upper()

    if basename.startswith('WHSE') or basename.startswith('REG'):
        return True

    if '.gdb' in datasource:
        fc_name = datasource.split('.gdb')[-1].strip('/\\').upper()
        if fc_name.startswith('WHSE') or fc_name.startswith('REG'):
            return True

    return False


def file_exists(datasource):
    """Check if file or GDB exists"""
    if '.gdb' in datasource:
        parts = datasource.split('.gdb')
        gdb_path = parts[0] + '.gdb'
        return os.path.exists(gdb_path)
    else:
        return os.path.exists(datasource)


def read_spatial_data(datasource):
    """Read spatial data from shp or GDB featureclass"""
    try:
        if datasource.lower().endswith('.shp'):
            try:
                gdf = gpd.read_file(datasource)
            except Exception as e:
                print(f"    Warning: pyogrio failed ({str(e)}). Attempting with fiona...")
                gdf = gpd.read_file(datasource, engine='fiona')

        elif '.gdb' in datasource:
            parts = datasource.split('.gdb')
            gdb = parts[0] + '.gdb'
            fc = os.path.basename(datasource)

            try:
                gdf = gpd.read_file(filename=gdb, layer=fc)
            except Exception as e:
                print(f"    Warning: pyogrio failed ({str(e)}). Attempting with fiona...")
                gdf = gpd.read_file(filename=gdb, layer=fc, engine='fiona')
        else:
            raise Exception(f'Format not recognized: {datasource}')

        return gdf

    except Exception as e:
        raise Exception(f"Failed to read {datasource}: {str(e)}")


def load_spatial_data(datasource, schema, table_name, file_modified_date):
    """Load spatial data into PostGIS"""
    try:
        print(f"  Loading {datasource} into {schema}.{table_name}...")

        # Read spatial data
        gdf = read_spatial_data(datasource)

        # Ensure geometry column is named 'geometry'
        if gdf.geometry.name != 'geometry':
            gdf = gdf.rename_geometry('geometry')

        # Convert to EPSG:3005 (BC Albers) if not already
        if gdf.crs and gdf.crs.to_epsg() != 3005:
            print(f"    Reprojecting from {gdf.crs.to_epsg()} to EPSG:3005...")
            gdf = gdf.to_crs(epsg=3005)

        feature_count = len(gdf)

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
            index_name = f"sidx_{table_name}_geom"[:63]
            sql = text(f'CREATE INDEX IF NOT EXISTS {index_name} ON {schema}.{table_name} USING GIST (geometry);')
            conn.execute(sql)
            conn.commit()

        # Update metadata
        update_dataset_metadata(schema, table_name, datasource, file_modified_date, feature_count)

        print(f"    ✓ Successfully loaded {feature_count} features with spatial index")
        return True

    except Exception as e:
        print(f"    ✗ Error loading dataset: {str(e)}")
        return False


# Process each region
for schema, excel_file in in_files.items():
    print(f"\n{'='*60}")
    print(f"Processing {schema.upper()} region: {excel_file}")
    print('='*60)

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
        datasource = str(datasource).strip()
        datasource = clean_path(datasource)
        datasource = convert_to_mapped_drive(datasource)

        # Skip WHSE/REG datasets
        if should_skip_dataset(datasource):
            continue

        # Check for duplicates in spreadsheet
        if datasource in processed_in_spreadsheet:
            print(f"Skipping {datasource} (duplicate in same spreadsheet)")
            skipped_datasets.append(f"{schema}: {datasource} (duplicate in spreadsheet)")
            continue

        # Check if file exists
        if not file_exists(datasource):
            print(f"Skipping {datasource} (file not found)")
            errors.append(f"{schema}: {datasource} (file not found)")
            continue

        # Get table name
        table_name = get_table_name(datasource)

        # Get file modified date
        file_modified_date = get_file_modified_date(datasource)

        # Check if update is needed
        print(f"Processing: {os.path.basename(datasource)}")

        if not needs_update(schema, table_name, datasource, file_modified_date):
            unchanged_datasets.append(f"{schema}.{table_name}")
            processed_in_spreadsheet.add(datasource)
            continue

        # Load the data
        success = load_spatial_data(datasource, schema, table_name, file_modified_date)

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

print(f"\nLoaded/Updated: {len(loaded_datasets)} datasets")
for dataset in loaded_datasets:
    print(f"  ✓ {dataset}")

print(f"\nUnchanged (skipped): {len(unchanged_datasets)} datasets")
for dataset in unchanged_datasets:
    print(f"  = {dataset}")

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
