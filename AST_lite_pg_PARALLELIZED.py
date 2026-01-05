"""
Automatic Status Tool - LITE Version (PARALLELIZED)

Purpose:     This script checks for overlaps between an AOI and datasets
             specified in the AST datasets spreadsheets (common and region specific). 
             
Notes        The script supports AOIs in TANTALIS Crown Tenure spatial view 
             and User defined AOIs (shp, featureclass).
               
             The script generates a spreadhseet of conflicts (TAB3) of the 
             standard AST reportand Interactive HTML maps showing the AOI and ovelappng features

            This version of the script uses postgis to process local datasets.
                             
Arguments:   - Output location (workspace)
             - DB credentials for Oracle/BCGW and PostGIS
             - Input source: TANTALIS OR AOI
             - Region (west coast, skeena...)
             - AOI: - ESRI shp or featureclass (AOI) OR
                    - TANTALIS File number
                    - TANTALIS Disposition ID
                    - TANTALIS Parcel ID

Author: Moez Labiadh - GeoBC

Created: 2025-12-23
Updated: 2025-01-05
"""

import warnings
warnings.simplefilter(action='ignore')

import os
import re
import timeit
import oracledb
import psycopg2
import pandas as pd
import folium
import geopandas as gpd
from pathlib import Path
from shapely import from_wkt, wkb
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Thread-safe locks for shared resources
results_lock = Lock()
print_lock = Lock()

def thread_print(message):
    """Thread-safe print function"""
    with print_lock:
        print(message)


def connect_to_Oracle(username, password, hostname):
    """Returns a connection and cursor to Oracle database"""
    try:
        connection = oracledb.connect(user=username, password=password, dsn=hostname)
        cursor = connection.cursor()
        return connection, cursor
    except:
        raise Exception('....Connection failed! Please check your login parameters')


def connect_to_PostGIS(host, database, user, password, port=5432):
    """Returns a connection and cursor to PostGIS database"""
    try:
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        cursor = connection.cursor()
        return connection, cursor
    except Exception as e:
        raise Exception(f'....PostGIS connection failed! Error: {e}')


def read_query(connection, cursor, query, bvars):
    """Returns a df containing SQL Query results"""
    cursor.execute(query, bvars)
    names = [x[0] for x in cursor.description]
    rows = cursor.fetchall()
    
    # Convert any LOB objects to strings immediately
    processed_rows = []
    for row in rows:
        processed_row = []
        for val in row:
            # Check if it's a LOB object and convert to string
            if hasattr(val, 'read'):  # LOB objects have a read method
                try:
                    processed_row.append(str(val))
                except:
                    processed_row.append(val)
            else:
                processed_row.append(val)
        processed_rows.append(tuple(processed_row))
    
    df = pd.DataFrame(processed_rows, columns=names)
    return df


def esri_to_gdf(aoi):
    """Returns a Geopandas file (gdf) based on 
       an ESRI format vector (shp or featureclass/gdb)"""
    if '.shp' in aoi: 
        gdf = gpd.read_file(aoi)
    elif '.gdb' in aoi:
        l = aoi.split('.gdb')
        gdb = l[0] + '.gdb'
        fc = os.path.basename(aoi)
        gdf = gpd.read_file(filename=gdb, layer=fc)
    else:
        raise Exception('Format not recognized. Please provide a shp or featureclass (gdb)!')
    return gdf


def df_2_gdf(df, crs):
    """Return a geopandas gdf based on a df with Geometry column, handling curve geometries"""
    df_clean = pd.DataFrame()
    
    if 'SHAPE' in df.columns:
        shape_col = 'SHAPE'
    elif 'shape' in df.columns:
        shape_col = 'shape'
    else:
        raise ValueError("No geometry column found. Expected 'SHAPE' or 'shape'")
    
    # Handle both string and LOB data types
    # If already strings, just use them; otherwise convert
    if df[shape_col].dtype == 'object':
        # Check if first item is already a string
        first_val = df[shape_col].iloc[0]
        if isinstance(first_val, str):
            shape_series = df[shape_col]
        else:
            # It's a LOB object, convert to string
            shape_series = df[shape_col].apply(lambda x: str(x) if x is not None else None)
    else:
        shape_series = df[shape_col].astype(str)
    
    def linearize_geometry(wkt_str):
        """Convert curve geometries to linear approximations"""
        from osgeo import ogr
        try:
            geom = ogr.CreateGeometryFromWkt(wkt_str)
            if geom is None:
                return None
            linear_geom = geom.GetLinearGeometry()
            return linear_geom.ExportToWkt()
        except Exception as e:
            print(f"Error processing geometry: {e}")
            return None
    
    def process_geometry(wkt_str):
        """Only linearize if it contains curve geometry types"""
        if wkt_str is None or not isinstance(wkt_str, str):
            return None
        if any(curve_type in wkt_str.upper() for curve_type in ['CURVE', 'CIRCULARSTRING', 'COMPOUNDCURVE']):
            return linearize_geometry(wkt_str)
        return wkt_str
    
    processed_wkts = [process_geometry(wkt) for wkt in shape_series]
    
    for col in df.columns:
        if col not in [shape_col]:
            df_clean[col] = df[col].values
    
    df_clean['geometry'] = gpd.GeoSeries.from_wkt(processed_wkts, crs=f"EPSG:{crs}")
    gdf = gpd.GeoDataFrame(df_clean, geometry='geometry', crs=f"EPSG:{crs}")

    return gdf


def multipart_to_singlepart(gdf):
    """Converts a multipart gdf to singlepart gdf"""
    gdf['dissolvefield'] = 1
    gdf = gdf.dissolve(by='dissolvefield')
    gdf.reset_index(inplace=True)
    gdf = gdf[['geometry']]
    return gdf


def get_wkb_srid(gdf):
    """Returns SRID and WKB objects from gdf"""
    srid = gdf.crs.to_epsg()
    geom = gdf['geometry'].iloc[0]
    wkb_aoi = wkb.dumps(geom)
    
    if geom.has_z:
        wkb_aoi = wkb.dumps(geom, output_dimension=2)
    
    return wkb_aoi, srid


def get_table_name_from_datasource(datasource):
    """Extract table name from datasource path and clean it for PostgreSQL"""
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
        table_name = table_name[:50].rstrip('_')
    
    return table_name


def read_input_spreadsheets(wksp_xls, region):
    """Returns input spreadsheets"""
    common_xls = os.path.join(wksp_xls, 'one_status_common_datasets.xlsx')
    region_xls = os.path.join(wksp_xls, 'one_status_{}_specific.xlsx'.format(region.lower()))
    
    df_stat_c = pd.read_excel(common_xls)
    df_stat_r = pd.read_excel(region_xls)
    
    df_stat = pd.concat([df_stat_c, df_stat_r])
    df_stat.dropna(how='all', inplace=True)
    df_stat = df_stat.reset_index(drop=True)
    
    return df_stat


def get_table_cols(item_index, df_stat):
    """Returns table and field names from the AST datasets spreadsheet"""
    df_stat_item = df_stat.loc[[item_index]]
    df_stat_item.fillna(value='nan', inplace=True)

    table = df_stat_item['Datasource'].iloc[0].strip()
    
    fields = []
    
    first_field = str(df_stat_item['Fields_to_Summarize'].iloc[0].strip())
    if first_field != 'nan':
        fields.append(first_field)

    for f in range(2, 7):
        for i in df_stat_item['Fields_to_Summarize' + str(f)].tolist():
            if i != 'nan':
                fields.append(str(i.strip()))

    col_lbl = df_stat_item['map_label_field'].iloc[0].strip()
    
    if col_lbl != 'nan' and col_lbl not in fields:
        fields.append(col_lbl)
    
    if fields:
        cols = ','.join(fields)
    else:
        cols = ''
    
    return table, cols, col_lbl


def get_def_query(item_index, df_stat, for_postgis=False):
    """Returns a SQL formatted def query (if any) from the AST datasets spreadsheet"""
    df_stat_item = df_stat.loc[[item_index]]
    df_stat_item.fillna(value='nan', inplace=True)

    def_query = df_stat_item['Definition_Query'].iloc[0].strip()
    def_query = def_query.strip()
    
    if def_query == 'nan':
            def_query = " "
    else:
        def_query = def_query.replace('"', '')
        if for_postgis:
            def_query = def_query.replace('%', '%%')
        
        def_query = 'AND (' + def_query + ')'
    
    return def_query


def get_radius(item_index, df_stat):
    """Returns the buffer distance (if any) from the AST common datasets spreadsheet"""
    df_stat_item = df_stat.loc[[item_index]]
    df_stat_item.fillna(value=0, inplace=True)
    df_stat_item['Buffer_Distance'] = df_stat_item['Buffer_Distance'].astype(int)
    radius = df_stat_item['Buffer_Distance'].iloc[0]
    return radius


def filter_invalid_oracle_geom(query, table_name, geom_col):
    """Apply geometry validation filter for specific problematic Oracle tables"""
    PROBLEMATIC_TABLES = [
        'WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW',
        'WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_SVW'
    ]

    if table_name not in PROBLEMATIC_TABLES:
        return query

    validation_clause = (
        f"SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT({geom_col}, 0.5) = 'TRUE'\n  AND "
    )

    query = query.replace(
        'WHERE SDO_WITHIN_DISTANCE',
        f'WHERE {validation_clause}SDO_WITHIN_DISTANCE'
    )

    return query


def apply_coordinate_transform(query, geom_col, srid_t):
    """Apply coordinate transformation to Oracle spatial query when SRIDs don't match"""
    
    query = query.replace(
        f'SDO_GEOMETRY(:wkb_aoi, :srid), 0.5)',
        f'SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t), 0.5)'
    )
    
    query = query.replace(
        f'SDO_GEOMETRY(:wkb_aoi, :srid),',
        f'SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t),'
    )
    
    query = query.replace(
        f'SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE',
        f'SDO_UTIL.TO_WKTGEOMETRY(SDO_CS.TRANSFORM({geom_col}, :srid_t, :srid)) SHAPE'
    )
    
    return query


def load_queries():
    """Load SQL queries for Oracle and PostGIS"""
    sql = {}

    sql['aoi'] = """
                    SELECT SDO_UTIL.TO_WKTGEOMETRY(a.SHAPE) SHAPE
                    FROM  WHSE_TANTALIS.TA_CROWN_TENURES_SVW a
                    WHERE a.CROWN_LANDS_FILE = :file_nbr
                        AND a.DISPOSITION_TRANSACTION_SID = :disp_id
                        AND a.INTRID_SID = :parcel_id
                  """
                        
    sql['geomCol'] = """
                    SELECT column_name GEOM_NAME
                    FROM  ALL_SDO_GEOM_METADATA
                    WHERE owner = :owner
                        AND table_name = :tab_name
                    """    
                    
    sql['srid'] = """
                    SELECT s.{geom_col}.sdo_srid SP_REF
                    FROM {tab} s
                    WHERE rownum = 1
                   """
                                 
    sql['oracle_overlay'] = """
                    SELECT {cols},
                           CASE WHEN SDO_GEOM.SDO_DISTANCE({geom_col}, SDO_GEOMETRY(:wkb_aoi, :srid), 0.5) = 0 
                            THEN 'INTERSECT' 
                             ELSE 'Within ' || TO_CHAR({radius}) || ' m'
                              END AS RESULT,
                           SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE
                    FROM {tab}
                    WHERE SDO_WITHIN_DISTANCE ({geom_col}, 
                                               SDO_GEOMETRY(:wkb_aoi, :srid),'distance = {radius}') = 'TRUE'
                        {def_query}   
                                  """ 
    
    sql['postgis_overlay'] = """
                    SELECT {cols},
                           CASE 
                               WHEN ST_Intersects(geometry, ST_GeomFromWKB(%s, %s)) 
                               THEN 'INTERSECT'
                               ELSE 'Within ' || %s || ' m'
                           END AS result,
                           ST_AsText(geometry) AS shape
                    FROM {schema}.{table}
                    WHERE ST_DWithin(geometry, ST_GeomFromWKB(%s, %s), %s)
                    {def_query}
                    """
    
    return sql


def get_geom_colname(connection, cursor, table, geomQuery):
    """Returns the geometry column of BCGW table name: can be either SHAPE or GEOMETRY"""
    el_list = table.split('.')
    bvars_geom = {'owner': el_list[0].strip(), 'tab_name': el_list[1].strip()}
    df_g = read_query(connection, cursor, geomQuery, bvars_geom)
    geom_col = df_g['GEOM_NAME'].iloc[0]
    return geom_col


def get_geom_srid(connection, cursor, table, geom_col, sridQuery):
    """Returns the SRID of the BCGW table, or None if table is empty"""
    try:
        sridQuery = sridQuery.format(tab=table, geom_col=geom_col)
        df_s = read_query(connection, cursor, sridQuery, {})
        
        if df_s.empty or df_s.shape[0] == 0:
            return None
            
        srid_t = df_s['SP_REF'].iloc[0]
        return srid_t
        
    except IndexError:
        return None
    except Exception as e:
        return None


def get_oracle_columns(connection, cursor, table):
    """Retrieves the list of available columns for an Oracle table"""
    try:
        query = """
            SELECT column_name 
            FROM all_tab_columns 
            WHERE owner = :owner 
                AND table_name = :tab_name
        """
        
        el_list = table.split('.')
        bvars = {'owner': el_list[0].strip(), 'tab_name': el_list[1].strip()}
        df_cols = read_query(connection, cursor, query, bvars)
        
        if df_cols.empty:
            return []
            
        return df_cols['COLUMN_NAME'].tolist()
        
    except Exception as e:
        return []


def check_postgis_table_exists(pg_cursor, schema, table):
    """Check if a PostGIS table exists"""
    try:
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s 
                    AND table_name = %s
            )
        """
        pg_cursor.execute(query, (schema, table))
        exists = pg_cursor.fetchone()[0]
        return exists
    except Exception as e:
        return False


def get_postgis_columns(pg_connection, pg_cursor, schema, table):
    """Retrieves the list of available columns for a PostGIS table"""
    try:
        query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = %s 
                AND table_name = %s
                AND column_name != 'geometry'
        """
        
        pg_cursor.execute(query, (schema, table))
        columns = [row[0] for row in pg_cursor.fetchall()]
        return columns
        
    except Exception as e:
        return []


def convert_columns_to_uppercase(cols_str):
    """Convert comma-separated column names to uppercase for Oracle"""
    if not cols_str or cols_str == '':
        return ''
    
    cols_list = [c.strip().upper() for c in cols_str.split(',')]
    return ','.join(cols_list)


def convert_columns_to_lowercase(cols_str):
    """Convert comma-separated column names to lowercase for PostGIS"""
    if not cols_str or cols_str == '':
        return ''
    
    cols_list = [c.strip().lower() for c in cols_str.split(',')]
    return ','.join(cols_list)


def validate_columns(cols, available_cols, item, table, is_postgis=False):
    """
    Validates that requested columns exist in the dataset
    Handles both PostGIS (lowercase) and Oracle (uppercase)
    """
    missing_cols = []
    
    if isinstance(cols, str) and cols:
        requested = [c.strip() for c in cols.split(',')]
    else:
        requested = []
    
    if is_postgis:
        requested = [c.lower() for c in requested]
        available_cols = [c.lower() for c in available_cols]
        objectid_col = 'objectid'
        excluded_cols = ['shape', 'geometry']
    else:
        requested = [c.upper() for c in requested]
        available_cols = [c.upper() for c in available_cols]
        objectid_col = 'OBJECTID'
        excluded_cols = ['SHAPE', 'GEOMETRY']
    
    for col in requested:
        if col not in available_cols and col != objectid_col:
            missing_cols.append(col)
    
    if len(missing_cols) == len(requested) or not requested:
        if objectid_col in available_cols:
            return objectid_col, missing_cols
        
        valid_available = [col for col in available_cols if col not in excluded_cols]
        
        if valid_available:
            return valid_available[0], missing_cols
        else:
            return objectid_col, missing_cols
    
    valid_cols = [c for c in requested if c not in missing_cols]
    
    validated = ','.join(valid_cols) if valid_cols else objectid_col
    
    return validated, missing_cols


def simplify_geometries(gdf, tol=10, preserve_topology=True):
    """Simplifies geometries in a gdf for webmap display"""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=tol, preserve_topology=preserve_topology)
    return gdf


def make_status_map(gdf_aoi, gdf_intr, col_lbl, item, workspace):
    """Generates HTML Interactive maps of AOI and intersection geodataframes"""
    m = folium.Map(tiles='openstreetmap')
    xmin, ymin, xmax, ymax = gdf_aoi.to_crs(4326)['geometry'].total_bounds
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])

    gdf_aoi.explore(
         m=m,
         tooltip=False,
         style_kwds=dict(fill=False, color="red", weight=3),
         name="AOI")

    gdf_intr.explore(
         m=m,
         column=col_lbl, 
         tooltip=col_lbl, 
         popup=True, 
         cmap="Dark2",  
         style_kwds=dict(color="gray"),
         name=item)
    
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    maps_dir = os.path.join(workspace, 'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
        
    out_html = os.path.join(maps_dir, item + '.html')
    m.save(out_html)


def write_xlsx(results, df_stat, workspace):
    """Writes results to a spreadsheet"""
    df_res = df_stat[['Category', 'Featureclass_Name(valid characters only)']]   
    df_res.rename(columns={'Featureclass_Name(valid characters only)': 'item'}, inplace=True)
    df_res['List of conflicts'] = ""
    df_res['Map'] = ""
    
    expanded_rows = []
    
    for index, row in df_res.iterrows():
        has_conflicts = False
        for k, v in results.items():
            if row['item'] == k and v.shape[0] > 0:
                has_conflicts = True
                
                result_col_to_drop = None
                if 'RESULT' in v.columns:
                    result_col_to_drop = 'RESULT'
                elif 'result' in v.columns:
                    result_col_to_drop = 'result'
                
                if result_col_to_drop:
                    v = v.drop(result_col_to_drop, axis=1)
                
                v['Result'] = v[v.columns].apply(lambda row: '; '.join(row.values.astype(str)), axis=1)
                
                for conflict in v['Result'].to_list():
                    expanded_rows.append({
                        'Category': row['Category'],
                        'item': row['item'],
                        'List of conflicts': str(conflict),
                        'Map': '=HYPERLINK("{}", "View Map")'.format(os.path.join(workspace, 'maps', k + '.html'))
                    })
                break
        
        if not has_conflicts:
            expanded_rows.append({
                'Category': row['Category'],
                'item': row['item'],
                'List of conflicts': '',
                'Map': ''
            })
    
    df_res = pd.DataFrame(expanded_rows)

    filename = os.path.join(workspace, 'AST_lite_TAB3.xlsx')
    sheetname = 'Conflicts & Constraints'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')        
    df_res.to_excel(writer, sheet_name=sheetname, index=False, startrow=0, startcol=0)
    
    workbook = writer.book
    worksheet = writer.sheets[sheetname]
    
    txt_format = workbook.add_format({'text_wrap': True})
    lnk_format = workbook.add_format({'underline': True, 'font_color': 'blue'})
    worksheet.set_column(0, 0, 30)
    worksheet.set_column(1, 1, 60)
    worksheet.set_column(2, 2, 80, txt_format)
    worksheet.set_column(3, 3, 20)
    
    worksheet.conditional_format('D2:D{}'.format(df_res.shape[0] + 1), 
                                 {'type': 'cell',
                                  'criteria': 'equal to', 
                                  'value': '"View Map"',
                                  'format': lnk_format})
    
    col_names = [{'header': col_name} for col_name in df_res.columns]
    worksheet.add_table(0, 0, df_res.shape[0] + 1, df_res.shape[1] - 1, {'columns': col_names})
    
    writer.close()


def process_single_dataset(args):
    """
    Process a single dataset - designed to run in parallel
    Each thread gets its own database connections
    """
    (index, row, df_stat, sql, wkb_aoi, srid, region, out_wksp, gdf_aoi,
     bcgw_user, bcgw_pwd, hostname, pg_host, pg_database, pg_user, pg_pwd, pg_port) = args
    
    item = row['Featureclass_Name(valid characters only)']
    
    # Create thread-local database connections
    connection = None
    cursor = None
    pg_connection = None
    pg_cursor = None
    
    try:
        thread_print(f'🔄 Processing: {item}')
        
        # Get dataset parameters
        table, cols, col_lbl = get_table_cols(index, df_stat)
        def_query = get_def_query(index, df_stat, for_postgis=False)
        radius = get_radius(index, df_stat)
        
        validated_cols = cols
        df_all = None
        df_all_res = None
        gdf_intr = None
        col_lbl_final = None
        ov_nbr = 0
        
        # Determine if this is Oracle or PostGIS
        if table.startswith('WHSE') or table.startswith('REG'):
            # ORACLE DATASET
            connection, cursor = connect_to_Oracle(bcgw_user, bcgw_pwd, hostname)
            
            geomQuery = sql['geomCol']
            sridQuery = sql['srid']
            geom_col = get_geom_colname(connection, cursor, table, geomQuery)
            
            srid_t = get_geom_srid(connection, cursor, table, geom_col, sridQuery)
            
            if srid_t is None:
                thread_print(f'⚠️  {item}: Table is empty - SKIPPED')
                return {
                    'item': item,
                    'df_result': pd.DataFrame([]),
                    'gdf_intr': None,
                    'col_lbl': None,
                    'overlaps': 0,
                    'success': False,
                    'error': 'Empty table'
                }
            
            available_cols = get_oracle_columns(connection, cursor, table)
            
            if not available_cols:
                thread_print(f'⚠️  {item}: Could not retrieve columns - SKIPPED')
                return {
                    'item': item,
                    'df_result': pd.DataFrame([]),
                    'gdf_intr': None,
                    'col_lbl': None,
                    'overlaps': 0,
                    'success': False,
                    'error': 'Could not retrieve table columns'
                }
            
            cols_uppercase = convert_columns_to_uppercase(cols)
            col_lbl_uppercase = col_lbl.upper() if col_lbl != 'nan' else 'nan'
            
            validated_cols, missing_cols = validate_columns(cols_uppercase, available_cols, item, table, is_postgis=False)
            
            if missing_cols:
                thread_print(f'⚠️  {item}: Missing columns: {", ".join(missing_cols[:3])}{"..." if len(missing_cols) > 3 else ""}')
            
            query = sql['oracle_overlay'].format(
                cols=validated_cols, tab=table, radius=radius,
                geom_col=geom_col, def_query=def_query)
            
            srid_mismatch = (srid_t == 1000003005)
            
            if srid_mismatch:
                query = apply_coordinate_transform(query, geom_col, srid_t)
                cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
                bvars_intr = {'wkb_aoi': wkb_aoi, 'srid': int(srid), 'srid_t': int(srid_t)}
            else:
                cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
                bvars_intr = {'wkb_aoi': wkb_aoi, 'srid': int(srid)}
            
            query = filter_invalid_oracle_geom(query, table, geom_col)
            
            df_all = read_query(connection, cursor, query, bvars_intr)
            
            result_col = 'RESULT'
            col_lbl_to_use = col_lbl_uppercase if col_lbl != 'nan' else 'nan'
            
        else:
            # POSTGIS DATASET
            pg_connection, pg_cursor = connect_to_PostGIS(pg_host, pg_database, pg_user, pg_pwd, pg_port)
            
            table_name = get_table_name_from_datasource(table)
            schema = region.lower()
            
            table_exists = check_postgis_table_exists(pg_cursor, schema, table_name)
            
            if not table_exists:
                thread_print(f'⚠️  {item}: Table not found in PostGIS - SKIPPED')
                return {
                    'item': item,
                    'df_result': pd.DataFrame([]),
                    'gdf_intr': None,
                    'col_lbl': None,
                    'overlaps': 0,
                    'success': False,
                    'error': f'Table not found in PostGIS: {schema}.{table_name}'
                }
            
            available_cols = get_postgis_columns(pg_connection, pg_cursor, schema, table_name)
            
            if not available_cols:
                thread_print(f'⚠️  {item}: No columns found - SKIPPED')
                return {
                    'item': item,
                    'df_result': pd.DataFrame([]),
                    'gdf_intr': None,
                    'col_lbl': None,
                    'overlaps': 0,
                    'success': False,
                    'error': f'No columns found in PostGIS table {schema}.{table_name}'
                }

            cols_lowercase = convert_columns_to_lowercase(cols)
            col_lbl_lowercase = col_lbl.lower() if col_lbl != 'nan' else 'nan'
            
            validated_cols, missing_cols = validate_columns(cols_lowercase, available_cols, item, table_name, is_postgis=True)
            
            if missing_cols:
                thread_print(f'⚠️  {item}: Missing columns: {", ".join(missing_cols[:3])}{"..." if len(missing_cols) > 3 else ""}')
            
            pg_def_query = get_def_query(index, df_stat, for_postgis=True)
            
            query = sql['postgis_overlay'].format(
                cols=validated_cols,
                schema=schema,
                table=table_name,
                def_query=pg_def_query
            )

            pg_cursor.execute(query, (
                psycopg2.Binary(wkb_aoi), 
                int(srid), 
                str(radius),
                psycopg2.Binary(wkb_aoi), 
                int(srid), 
                float(radius)
            ))
            
            names = [desc[0] for desc in pg_cursor.description]
            rows = pg_cursor.fetchall()
            df_all = pd.DataFrame(rows, columns=names)
            
            result_col = 'result'
            col_lbl_to_use = col_lbl_lowercase if col_lbl != 'nan' else 'nan'
        
        # Process results (common for both Oracle and PostGIS)
        if isinstance(validated_cols, str):
            cols_list = [c.strip() for c in validated_cols.split(",")]
        else:
            cols_list = validated_cols

        if result_col in df_all.columns:
            cols_list.append(result_col)

        available_result_cols = [col for col in cols_list if col in df_all.columns]

        if not available_result_cols:
            thread_print(f'⚠️  {item}: No valid columns in results - SKIPPED')
            return {
                'item': item,
                'df_result': pd.DataFrame([]),
                'gdf_intr': None,
                'col_lbl': None,
                'overlaps': 0,
                'success': True,
                'error': None
            }

        df_all_res = df_all[available_result_cols]
        ov_nbr = df_all_res.shape[0]

        if ov_nbr > 0:
            thread_print(f'✅ {item}: {ov_nbr} overlap{"s" if ov_nbr != 1 else ""} found')
            gdf_intr = df_2_gdf(df_all, 3005)
            
            if col_lbl_to_use == 'nan': 
                col_lbl_final = cols_list[0]
            else:
                col_lbl_final = col_lbl_to_use
            
            if col_lbl_final in gdf_intr.columns:
                gdf_intr[col_lbl_final] = gdf_intr[col_lbl_final].astype(str)
            else:
                col_lbl_final = cols_list[0]
                if col_lbl_final in gdf_intr.columns:
                    gdf_intr[col_lbl_final] = gdf_intr[col_lbl_final].astype(str)
            
            for col in gdf_intr.columns:
                if gdf_intr[col].dtype == 'datetime64[ns]':
                    gdf_intr[col] = gdf_intr[col].astype(str)
        else:
            thread_print(f'✅ {item}: No overlaps')
        
        return {
            'item': item,
            'df_result': df_all_res,
            'gdf_intr': gdf_intr,
            'col_lbl': col_lbl_final,
            'overlaps': ov_nbr,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        thread_print(f'❌ {item}: ERROR - {str(e)[:100]}')
        return {
            'item': item,
            'df_result': pd.DataFrame([]),
            'gdf_intr': None,
            'col_lbl': None,
            'overlaps': 0,
            'success': False,
            'error': str(e)
        }
    
    finally:
        # Clean up connections
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        if pg_cursor:
            pg_cursor.close()
        if pg_connection:
            pg_connection.close()


if __name__ == "__main__":
    """Executes the AST light process with parallelization"""
    start_t = timeit.default_timer()
    
    # Paths
    workspace = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
    wksp_xls = os.path.join(workspace, 'input_spreadsheets')
    aoi = os.path.join(workspace, 'test_data', 'aoi_test_4.shp')
    out_wksp = os.path.join(workspace, 'outputs')
    
    # Database connection parameters
    hostname = 'bcgw.bcgov/idwprod1.bcgov'
    bcgw_user = os.getenv('bcgw_user')
    bcgw_pwd = os.getenv('bcgw_pwd')
    
    pg_host = 'localhost'
    pg_database = 'ast_local_datasets'
    pg_user = 'postgres'
    pg_pwd = os.getenv('PG_LCL_SUSR_PASS')
    pg_port = 5432
    
    # Test connections first
    print('Testing BCGW connection.')
    test_conn, test_cursor = connect_to_Oracle(bcgw_user, bcgw_pwd, hostname)
    test_cursor.close()
    test_conn.close()
    print("....Successfully connected to Oracle database")
    
    print('\nTesting PostGIS connection.')
    test_pg_conn, test_pg_cursor = connect_to_PostGIS(pg_host, pg_database, pg_user, pg_pwd, pg_port)
    test_pg_cursor.close()
    test_pg_conn.close()
    print("....Successfully connected to PostGIS database")
    
    print('\nLoading SQL queries')
    sql = load_queries()
    
    print('\nReading User inputs: AOI.')
    input_src = 'TANTALIS'  ####### USER INPUT ####### Possible values are "TANTALIS" and AOI

    if input_src == 'AOI':
        print('....Reading the AOI file')
        gdf_aoi = esri_to_gdf(aoi)
       
    elif input_src == 'TANTALIS':
        fileNbr = '8015096'
        dispID = 953116
        prclID = 989206
     
        in_fileNbr = fileNbr
        in_dispID = dispID
        in_prclID = prclID
        print('....input File Number: {}'.format(in_fileNbr))
        print('....input Disposition ID: {}'.format(in_dispID))
        print('....input Parcel ID: {}'.format(in_prclID))
        
        bvars_aoi = {'file_nbr': in_fileNbr, 'disp_id': in_dispID, 'parcel_id': in_prclID}
        
        print('....Querying TANTALIS for AOI geometry')
        # Create temporary connection for AOI
        temp_conn, temp_cursor = connect_to_Oracle(bcgw_user, bcgw_pwd, hostname)
        df_aoi = read_query(temp_conn, temp_cursor, sql['aoi'], bvars_aoi)
        
        if df_aoi.shape[0] < 1:
            temp_cursor.close()
            temp_conn.close()
            raise Exception('Parcel not in TANTALIS. Please check inputs!')
        else:
            print('....Converting TANTALIS result to GeoDataFrame')
            # Force LOB data to be read while connection is still open
            if 'SHAPE' in df_aoi.columns:
                df_aoi['SHAPE'] = df_aoi['SHAPE'].apply(lambda x: str(x) if x is not None else None)
            
            gdf_aoi = df_2_gdf(df_aoi, 3005)
            
            # Now safe to close connection
            temp_cursor.close()
            temp_conn.close()
               
    else:
        raise Exception('Possible input sources are TANTALIS and AOI!')
    
    if gdf_aoi.shape[0] > 1:
        print('....Converting multipart AOI to singlepart AOI')
        gdf_aoi = multipart_to_singlepart(gdf_aoi)
    
    print('....Extracting WKB and SRID from AOI')
    wkb_aoi, srid = get_wkb_srid(gdf_aoi)
    
    print('\nReading the AST datasets spreadsheet.')
    region = 'northeast'   ####### USER INPUT #######
    print('....Region is {}'.format(region))
    df_stat = read_input_spreadsheets(wksp_xls, region)
    
    print('\n' + '='*80)
    print('RUNNING PARALLEL ANALYSIS')
    print('='*80)
    
    results = {}
    failed_datasets = []
    datasets_with_overlaps = []
    
    item_count = df_stat.shape[0]
    
    # Prepare arguments for each dataset
    dataset_args = []
    for index, row in df_stat.iterrows():
        args = (index, row, df_stat, sql, wkb_aoi, srid, region, out_wksp, gdf_aoi,
                bcgw_user, bcgw_pwd, hostname, pg_host, pg_database, pg_user, pg_pwd, pg_port)
        dataset_args.append(args)
    
    # Configure parallelization
    max_workers = 4  ####### USER INPUT ####### Adjust based on your system (recommended: 3-6)
    print(f'\nConfiguration:')
    print(f'  - Parallel workers: {max_workers}')
    print(f'  - Total datasets: {item_count}')
    print(f'  - Region: {region}')
    print('\n' + '='*80)
    print('PROCESSING DATASETS')
    print('='*80 + '\n')
    
    # Process datasets in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(process_single_dataset, args): args[1]['Featureclass_Name(valid characters only)'] 
            for args in dataset_args
        }
        
        # Process results as they complete
        completed = 0
        start_time = timeit.default_timer()
        
        for future in as_completed(future_to_dataset):
            completed += 1
            dataset_name = future_to_dataset[future]
            
            try:
                result = future.result()
                
                # Thread-safe result storage
                with results_lock:
                    results[result['item']] = result['df_result']
                    
                    if not result['success']:
                        failed_datasets.append({
                            'item': result['item'], 
                            'reason': result['error']
                        })
                    elif result['gdf_intr'] is not None and result['overlaps'] > 0:
                        # Generate map
                        gdf_intr_s = simplify_geometries(result['gdf_intr'], tol=10, preserve_topology=True)
                        make_status_map(gdf_aoi, gdf_intr_s, result['col_lbl'], result['item'], out_wksp)
                        datasets_with_overlaps.append({
                            'item': result['item'],
                            'overlaps': result['overlaps']
                        })
                
                # Progress update
                elapsed = timeit.default_timer() - start_time
                avg_time = elapsed / completed
                remaining = (item_count - completed) * avg_time
                
                progress_bar = '█' * int(50 * completed / item_count) + '░' * (50 - int(50 * completed / item_count))
                
                with print_lock:
                    print(f'\n[{progress_bar}] {completed}/{item_count} ({completed*100//item_count}%)')
                    print(f'   Time elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | '
                          f'Est. remaining: {int(remaining//60)}m {int(remaining%60)}s')
                
            except Exception as e:
                thread_print(f'❌ Critical error processing {dataset_name}: {e}')
                with results_lock:
                    failed_datasets.append({'item': dataset_name, 'reason': str(e)})
    
    print('\n' + '='*80)
    print('WRITING RESULTS TO SPREADSHEET')
    print('='*80)
    write_xlsx(results, df_stat, out_wksp)
    print('✅ Results spreadsheet created')
    
    # Print comprehensive summary
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins = int(t_sec / 60)
    secs = int(t_sec % 60)
    
    print('\n' + '='*80)
    print('PROCESSING SUMMARY')
    print('='*80)
    print(f'\n📊 Statistics:')
    print(f'   Total datasets processed: {item_count}')
    print(f'   Datasets with overlaps: {len(datasets_with_overlaps)}')
    print(f'   Datasets with no overlaps: {item_count - len(datasets_with_overlaps) - len(failed_datasets)}')
    print(f'   Failed datasets: {len(failed_datasets)}')
    
    print(f'\n⏱️  Performance:')
    print(f'   Total time: {mins}m {secs}s')
    print(f'   Average per dataset: {t_sec/item_count:.1f}s')
    print(f'   Parallel workers used: {max_workers}')
    
    if datasets_with_overlaps:
        print(f'\n🔍 Datasets with overlaps:')
        for ds in sorted(datasets_with_overlaps, key=lambda x: x['overlaps'], reverse=True)[:10]:
            print(f'   • {ds["item"]}: {ds["overlaps"]} overlap(s)')
        if len(datasets_with_overlaps) > 10:
            print(f'   ... and {len(datasets_with_overlaps) - 10} more')
    
    if failed_datasets:
        print(f'\n❌ Failed datasets:')
        for failed in failed_datasets[:10]:
            print(f'   • {failed["item"]}')
            print(f'     Reason: {failed["reason"][:80]}{"..." if len(failed["reason"]) > 80 else ""}')
        if len(failed_datasets) > 10:
            print(f'   ... and {len(failed_datasets) - 10} more')
    else:
        print(f'\n✅ All datasets processed successfully!')
    
    print('\n' + '='*80)
    print(f'Output location: {out_wksp}')
    print('='*80)