"""
Name:        Automatic Status Tool - LITE version! DRAFT

Purpose:     This script checks for overlaps between an AOI and datasets
             specified in the AST datasets spreadsheets (common and region specific). 
             
Notes        The script supports AOIs in TANTALIS Crown Tenure spatial view 
             and User defined AOIs (shp, featureclass).
               
             The script generates a spreadhseet of conflicts and 
             Interactive HTML maps showing the AOI and ovelappng features

             This version uses PostGIS to process local datasets instead of Geopandas
                             
Arguments:   - Output location (workspace)
             - BCGW username
             - BCGW password
             - Region (west coast, skeena...)
             - AOI: - ESRI shp or featureclass OR
                    - File number
                    - Disposition ID
                    - Parcel ID
                
Author:      Moez Labiadh - GeoBC

Created:     2025-12-23
Updated:     2025-12-23
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



def connect_to_Oracle(username, password, hostname):
    """Returns a connection and cursor to Oracle database"""
    try:
        connection = oracledb.connect(user=username, password=password, dsn=hostname)
        cursor = connection.cursor()
        print("....Successfully connected to Oracle database")
    except:
        raise Exception('....Connection failed! Please check your login parameters')
    return connection, cursor



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
        print("....Successfully connected to PostGIS database")
    except Exception as e:
        raise Exception(f'....PostGIS connection failed! Error: {e}')
    return connection, cursor



def read_query(connection, cursor, query, bvars):
    """Returns a df containing SQL Query results"""
    cursor.execute(query, bvars)
    names = [x[0] for x in cursor.description]
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=names)
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
    
    # Handle both 'SHAPE' (Oracle) and 'shape' (PostGIS) column names
    if 'SHAPE' in df.columns:
        shape_col = 'SHAPE'
    elif 'shape' in df.columns:
        shape_col = 'shape'
    else:
        raise ValueError("No geometry column found. Expected 'SHAPE' or 'shape'")
    
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
        if col not in [shape_col]:  # Exclude the geometry column (either SHAPE or shape)
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
        original_length = len(table_name)
        table_name = table_name[:50].rstrip('_')
        print(f"    Note: Table name truncated from {original_length} to {len(table_name)} characters")
    
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
    fields.append(str(df_stat_item['Fields_to_Summarize'].iloc[0].strip()))


    for f in range(2, 7):
        for i in df_stat_item['Fields_to_Summarize' + str(f)].tolist():
            if i != 'nan':
                fields.append(str(i.strip()))


    col_lbl = df_stat_item['map_label_field'].iloc[0].strip()
    
    if col_lbl != 'nan' and col_lbl not in fields:
        fields.append(col_lbl)
    
    # Clean up fields list
    if fields[0] == 'nan':
        fields = []
    
    # Return as comma-separated string for consistency
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


    print('.......applying SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT filter (only TRUE geometries kept)')
    print('.......Note: All invalid geometries will be excluded from the overlay')


    validation_clause = (
        f"SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT({geom_col}, 0.5) = 'TRUE'\n  AND "
    )


    query = query.replace(
        'WHERE SDO_WITHIN_DISTANCE',
        f'WHERE {validation_clause}SDO_WITHIN_DISTANCE'
    )


    return query



def load_queries():
    """Load SQL queries for Oracle and PostGIS"""
    sql = {}


    # Oracle queries
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
                                 
    sql['overlay_wkb'] = """
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
    
    sql['overlay_wkb_transform'] = """
                    SELECT {cols},
                           CASE WHEN SDO_GEOM.SDO_DISTANCE({geom_col}, 
                                    SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t), 0.5) = 0 
                            THEN 'INTERSECT' 
                             ELSE 'Within ' || TO_CHAR({radius}) || ' m'
                              END AS RESULT,
                           SDO_UTIL.TO_WKTGEOMETRY(SDO_CS.TRANSFORM({geom_col}, :srid_t, :srid)) SHAPE
                    FROM {tab}
                    WHERE SDO_WITHIN_DISTANCE ({geom_col}, 
                                SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t),'distance = {radius}') = 'TRUE'
                        {def_query}   
                    """
    
    # PostGIS queries
    sql['postgis_overlay'] = """
                    SELECT {cols},
                           CASE 
                               WHEN ST_Intersects(geometry, ST_GeomFromWKB(%s, %s)) 
                               THEN 'INTERSECT'
                               ELSE 'Within ' || %s || ' m'
                           END AS RESULT,
                           ST_AsText(geometry) AS SHAPE
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
            print(f'.......WARNING: Table {table} is empty, cannot determine SRID')
            return None
            
        srid_t = df_s['SP_REF'].iloc[0]
        return srid_t
        
    except IndexError:
        print(f'.......WARNING: Table {table} appears to be empty, cannot determine SRID')
        return None
    except Exception as e:
        print(f'.......ERROR getting SRID for {table}: {e}')
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
        print(f'.......ERROR retrieving columns for {table}: {e}')
        return []



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
        print(f'.......ERROR retrieving PostGIS columns for {schema}.{table}: {e}')
        return []


def quote_postgres_columns(cols_str):
    """Add double quotes around PostgreSQL column names to preserve case"""
    if not cols_str or cols_str == '':
        return ''
    
    cols_list = [c.strip() for c in cols_str.split(',')]
    quoted_cols = [f'"{col}"' for col in cols_list]
    return ', '.join(quoted_cols)


def validate_columns(cols, available_cols, item, table):
    """Validates that requested columns exist in the dataset"""
    missing_cols = []
    
    # Parse comma-separated column string
    if isinstance(cols, str) and cols:
        requested = [c.strip() for c in cols.split(',')]
    else:
        requested = []
    
    # Check which columns are missing
    for col in requested:
        if col not in available_cols and col != 'OBJECTID':
            missing_cols.append(col)
    
    # If all columns are missing, fall back to first available column or OBJECTID
    if len(missing_cols) == len(requested) or not requested:
        if available_cols:
            print(f'.......WARNING: No valid requested columns, using {available_cols[0]}')
            return available_cols[0], missing_cols
        else:
            print(f'.......WARNING: No columns available, using OBJECTID')
            return 'OBJECTID', missing_cols
    
    # Remove missing columns from the list
    valid_cols = [c for c in requested if c not in missing_cols]
    
    if missing_cols:
        print(f'.......WARNING: Missing columns in {table}: {", ".join(missing_cols)}')
    
    # Return as comma-separated string
    validated = ','.join(valid_cols) if valid_cols else 'OBJECTID'
    
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
    
    folium.TileLayer('stamenterrain', control=True).add_to(m)
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
                
                # Handle both RESULT and result column names (Oracle vs PostGIS)
                if 'RESULT' in v.columns:
                    v = v.drop('RESULT', axis=1)
                elif 'result' in v.columns:
                    v = v.drop('result', axis=1)
                
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



if __name__ == "__main__":
    """Executes the AST light process"""
    start_t = timeit.default_timer()
    
    # Paths
    workspace = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
    wksp_xls = os.path.join(workspace, 'input_spreadsheets')
    aoi = os.path.join(workspace, 'test_data', 'aoi_test_4.shp')
    out_wksp = os.path.join(workspace, 'outputs')
    
    # Oracle connection
    print('Connecting to BCGW.')
    hostname = 'bcgw.bcgov/idwprod1.bcgov'
    bcgw_user = os.getenv('bcgw_user')
    bcgw_pwd = os.getenv('bcgw_pwd')
    connection, cursor = connect_to_Oracle(bcgw_user, bcgw_pwd, hostname)
    
    # PostGIS connection
    print('Connecting to PostGIS.')
    pg_host = 'localhost'
    pg_database = 'ast_local_datasets'
    pg_user = 'postgres'
    pg_pwd = os.getenv('PG_LCL_SUSR_PASS')
    pg_port = 5432
    pg_connection, pg_cursor = connect_to_PostGIS(pg_host, pg_database, pg_user, pg_pwd, pg_port)
    
    print('\nLoading SQL queries')
    sql = load_queries()
    
    print('\nReading User inputs: AOI.')
    input_src = 'TANTALIS'  # Possible values are "TANTALIS" and AOI


    if input_src == 'AOI':
        print('....Reading the AOI file')
        gdf_aoi = esri_to_gdf(aoi)
       
    elif input_src == 'TANTALIS':
        fileNbr = '1413717'
        dispID = 927132
        prclID = 953951
     
        in_fileNbr = fileNbr
        in_dispID = dispID
        in_prclID = prclID
        print('....input File Number: {}'.format(in_fileNbr))
        print('....input Disposition ID: {}'.format(in_dispID))
        print('....input Parcel ID: {}'.format(in_prclID))
        
        bvars_aoi = {'file_nbr': in_fileNbr, 'disp_id': in_dispID, 'parcel_id': in_prclID}
        
        print('....Querying TANTALIS for AOI geometry')
        df_aoi = read_query(connection, cursor, sql['aoi'], bvars_aoi) 
        
        if df_aoi.shape[0] < 1:
            raise Exception('Parcel not in TANTALIS. Please check inputs!')
        else:
            print('....Converting TANTALIS result to GeoDataFrame')
            gdf_aoi = df_2_gdf(df_aoi, 3005)
               
    else:
        raise Exception('Possible input sources are TANTALIS and AOI!')
    
    if gdf_aoi.shape[0] > 1:
        print('....Converting multipart AOI to singlepart AOI')
        gdf_aoi = multipart_to_singlepart(gdf_aoi)
    
    print('....Extracting WKB and SRID from AOI')
    wkb_aoi, srid = get_wkb_srid(gdf_aoi)
    
    print('\nReading the AST datasets spreadsheet.')
    region = 'west_coast'
    print('....Region is {}'.format(region))
    df_stat = read_input_spreadsheets(wksp_xls, region)
    
    print('\nRunning the analysis.')
    results = {}
    failed_datasets = []
    
    item_count = df_stat.shape[0]
    counter = 1
    
    for index, row in df_stat.iterrows():
        item = row['Featureclass_Name(valid characters only)']
        item_index = index
        
        print('\n****working on item {} of {}: {}***'.format(counter, item_count, item))
        
        try:
            print('.....getting table and column names')
            table, cols, col_lbl = get_table_cols(item_index, df_stat)
            
            print('.....getting definition query (if any)')
            def_query = get_def_query(item_index, df_stat, for_postgis=False)
        
            print('.....getting buffer distance (if any)')
            radius = get_radius(item_index, df_stat)  
            
            # This will hold the final validated columns
            validated_cols = cols
             
            print('.....running Overlay Analysis.')
            
            if table.startswith('WHSE') or table.startswith('REG'): 
                # Handle Oracle datasets
                geomQuery = sql['geomCol']
                sridQuery = sql['srid']
                geom_col = get_geom_colname(connection, cursor, table, geomQuery)
                
                srid_t = get_geom_srid(connection, cursor, table, geom_col, sridQuery)
                
                if srid_t is None:
                    print(f'.......SKIPPING dataset {item} - table is empty')
                    failed_datasets.append({'item': item, 'reason': 'Empty table'})
                    results[item] = pd.DataFrame([])
                    counter += 1
                    continue
                
                print('.....validating columns')
                available_cols = get_oracle_columns(connection, cursor, table)
                
                if not available_cols:
                    print(f'.......SKIPPING dataset {item} - could not retrieve table columns')
                    failed_datasets.append({'item': item, 'reason': 'Could not retrieve table columns'})
                    results[item] = pd.DataFrame([])
                    counter += 1
                    continue
                
                validated_cols, missing_cols = validate_columns(cols, available_cols, item, table)
                
                if missing_cols:
                    failed_datasets.append({'item': item, 'reason': f'Missing columns: {", ".join(missing_cols)}'})
                
                # NO b. prefix needed - columns come clean from database
                
                srid_mismatch = (srid_t == 1000003005)
                
                if srid_mismatch:
                    print(f'.......SRID mismatch detected (table SRID: {srid_t}), using transform query')
                    query = sql['overlay_wkb_transform'].format(
                        cols=validated_cols, tab=table, radius=radius,
                        geom_col=geom_col, def_query=def_query)
                    cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
                    bvars_intr = {'wkb_aoi': wkb_aoi, 'srid': int(srid), 'srid_t': int(srid_t)}
                else:
                    query = sql['overlay_wkb'].format(
                        cols=validated_cols, tab=table, radius=radius,
                        geom_col=geom_col, def_query=def_query)
                    cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
                    bvars_intr = {'wkb_aoi': wkb_aoi, 'srid': int(srid)}
                
                query = filter_invalid_oracle_geom(query, table, geom_col)
                df_all = read_query(connection, cursor, query, bvars_intr)
                    
            else:
                # Handle PostGIS datasets
                try:
                    # Get cleaned table name from datasource
                    table_name = get_table_name_from_datasource(table)
                    schema = region.lower()
                    
                    print(f'.......Using PostGIS table: {schema}.{table_name}')
                    
                    # Validate columns exist in PostGIS table
                    print('.....validating columns')
                    available_cols = get_postgis_columns(pg_connection, pg_cursor, schema, table_name)
                    
                    if not available_cols:
                        print(f'.......SKIPPING dataset {item} - could not retrieve PostGIS table columns')
                        failed_datasets.append({'item': item, 'reason': f'Could not retrieve PostGIS columns for {schema}.{table_name}'})
                        results[item] = pd.DataFrame([])
                        counter += 1
                        continue
                    
                    validated_cols, missing_cols = validate_columns(cols, available_cols, item, table_name)
                    
                    if missing_cols:
                        failed_datasets.append({'item': item, 'reason': f'Missing columns: {", ".join(missing_cols)}'})
                    
                    # IMPORTANT: Quote column names for PostgreSQL
                    quoted_cols = quote_postgres_columns(validated_cols)
                    
                    # Get definition query for PostGIS
                    pg_def_query = get_def_query(item_index, df_stat, for_postgis=True)
                    
                    # Build PostGIS overlay query using QUOTED columns
                    query = sql['postgis_overlay'].format(
                        cols=quoted_cols,  # Changed from validated_cols to quoted_cols
                        schema=schema,
                        table=table_name,
                        def_query=pg_def_query
                    )
                
                    # Execute PostGIS query
                    # Parameters: wkb_aoi, srid, radius_str, wkb_aoi, srid, radius
                    pg_cursor.execute(query, (
                        psycopg2.Binary(wkb_aoi), 
                        int(srid), 
                        str(radius),
                        psycopg2.Binary(wkb_aoi), 
                        int(srid), 
                        float(radius)
                    ))
                    
                    # Fetch results
                    names = [desc[0] for desc in pg_cursor.description]
                    rows = pg_cursor.fetchall()
                    df_all = pd.DataFrame(rows, columns=names)
                    
                except psycopg2.Error as e:
                    print(f'.......ERROR: PostGIS query failed: {e}')
                    failed_datasets.append({'item': item, 'reason': f'PostGIS query error: {str(e)}'})
                    results[item] = pd.DataFrame([])
                    counter += 1
                    continue
                except Exception as e:
                    print(f'.......ERROR: Could not process PostGIS dataset: {str(e)}')
                    failed_datasets.append({'item': item, 'reason': f'PostGIS dataset error: {str(e)}'})
                    results[item] = pd.DataFrame([])
                    counter += 1
                    continue
            
            # Process results (common for both Oracle and PostGIS)
            if isinstance(validated_cols, str):
                cols_list = [c.strip() for c in validated_cols.split(",")]
            else:
                cols_list = validated_cols

            # Handle RESULT column case sensitivity
            # Oracle returns uppercase, PostGIS returns lowercase
            if 'RESULT' in df_all.columns:
                cols_list.append('RESULT')
            elif 'result' in df_all.columns:
                cols_list.append('result')
            else:
                print(f'.......WARNING: Neither RESULT nor result column found in results')

            # Build the final column list, checking each column exists
            available_result_cols = [col for col in cols_list if col in df_all.columns]

            if not available_result_cols:
                print(f'.......WARNING: No valid columns found in results for {item}')
                results[item] = pd.DataFrame([])
                counter += 1
                continue

            df_all_res = df_all[available_result_cols] 
            
            ov_nbr = df_all_res.shape[0]
            print('.....number of overlaps: {}'.format(ov_nbr))
            
            results[item] = df_all_res

            if ov_nbr > 0:
                print('.....generating a map.')
                gdf_intr = df_2_gdf(df_all, 3005)
                
                if col_lbl == 'nan': 
                    col_lbl = cols_list[0]
                    gdf_intr[col_lbl] = gdf_intr[col_lbl].astype(str)
                
                for col in gdf_intr.columns:
                    if gdf_intr[col].dtype == 'datetime64[ns]':
                        gdf_intr[col] = gdf_intr[col].astype(str)
                
                gdf_intr[col_lbl] = gdf_intr[col_lbl].astype(str) 
                
                gdf_intr_s = simplify_geometries(gdf_intr, tol=10, preserve_topology=True)
                make_status_map(gdf_aoi, gdf_intr_s, col_lbl, item, out_wksp)

        except Exception as e:
            print(f'.......ERROR processing dataset {item}: {e}')
            failed_datasets.append({'item': item, 'reason': str(e)})
            results[item] = pd.DataFrame([])
        
        counter += 1
    
    print('\nWriting Results to spreadsheet')
    write_xlsx(results, df_stat, out_wksp)
    
    # Close PostGIS connection
    pg_cursor.close()
    pg_connection.close()
    
    # Print summary
    if failed_datasets:
        print('\n' + '='*80)
        print('SUMMARY: The following datasets failed to process:')
        print('='*80)
        for failed in failed_datasets:
            print(f"  - {failed['item']}")
            print(f"    Reason: {failed['reason']}")
        print(f'\nTotal failed datasets: {len(failed_datasets)} out of {item_count}')
        print('='*80)
    else:
        print('\n' + '='*80)
        print('SUCCESS: All datasets processed without errors!')
        print('='*80)
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins = int(t_sec / 60)
    secs = int(t_sec % 60)
    print('\nProcessing Completed in {} minutes and {} seconds'.format(mins, secs))