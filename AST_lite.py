"""
Name:        Automatic Status Tool - LITE version! DRAFT
Purpose:     This script checks for overlaps between an AOI and datasets
             specified in the AST datasets spreadsheets (common and region specific). 
             
Notes        The script supports AOIs in TANTALIS Crown Tenure spatial view 
             and User defined AOIs (shp, featureclass).
               
             The script generates a spreadhseet of conflicts and 
             Interactive HTML maps showing the AOI and ovelappng features
                             
Arguments:   - Output location (workspace)
             - BCGW username
             - BCGW password
             - Region (west coast, skeena...)
             - AOI: - ESRI shp or featureclass OR
                    - File number
                    - Disposition ID
                    - Parcel ID
                
Author:      Moez Labiadh

Created:     2023-01-12
Updated:     2025-12-15
"""



import warnings
warnings.simplefilter(action='ignore')

import os
import re
import timeit
import oracledb
import pandas as pd
import folium
import geopandas as gpd
from shapely import from_wkt, wkb
#from datetime import datetime



def connect_to_Oracle(username,password,hostname):
    """ Returns a connection and cursor to Oracle database"""
    try:
        connection = oracledb.connect(user=username, password=password, dsn=hostname)
        cursor = connection.cursor()
        print  ("....Successffuly connected to the database")
    except:
        raise Exception('....Connection failed! Please check your login parameters')

    return connection, cursor



def read_query(connection,cursor,query,bvars):
    "Returns a df containing SQL Query results"
    cursor.execute(query, bvars)
    names = [x[0] for x in cursor.description]
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=names)
    
    return df    
  

def esri_to_gdf (aoi):
    """Returns a Geopandas file (gdf) based on 
       an ESRI format vector (shp or featureclass/gdb)"""
    
    if '.shp' in aoi: 
        gdf = gpd.read_file(aoi)
    
    elif '.gdb' in aoi:
        l = aoi.split ('.gdb')
        gdb = l[0] + '.gdb'
        fc = os.path.basename(aoi)
        gdf = gpd.read_file(filename= gdb, layer= fc)
        
    else:
        raise Exception ('Format not recognized. Please provide a shp or featureclass (gdb)!')
    
    return gdf
           

def df_2_gdf(df, crs):
    """ Return a geopandas gdf based on a df with Geometry column, handling curve geometries"""
    df = df.copy()
    df['SHAPE'] = df['SHAPE'].astype(str)
    
    def linearize_geometry(wkt_str):
        """Convert curve geometries to linear approximations"""
        from osgeo import ogr
        try:
            # Create OGR geometry from WKT
            geom = ogr.CreateGeometryFromWkt(wkt_str)
            if geom is None:
                return None
            
            # GetLinearGeometry() converts curves to linear approximations
            linear_geom = geom.GetLinearGeometry()
            
            # Return as WKT that standard tools can handle
            return linear_geom.ExportToWkt()
        except Exception as e:
            print(f"Error processing geometry: {e}")
            return None
    
    def process_geometry(wkt_str):
        """Only linearize if it contains curve geometry types"""
        if any(curve_type in wkt_str.upper() for curve_type in ['CURVE', 'CIRCULARSTRING', 'COMPOUNDCURVE']):
            return linearize_geometry(wkt_str)
        return wkt_str
    
    # Only process geometries that need linearization
    df['processed_wkt'] = df['SHAPE'].apply(process_geometry)
    
    # Create geometries from the processed WKT
    df['geometry'] = gpd.GeoSeries.from_wkt(df['processed_wkt'])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=f"EPSG:{crs}")
    
    # Clean up temporary columns
    gdf = gdf.drop(columns=['SHAPE', 'processed_wkt'])
    
    return gdf



def multipart_to_singlepart(gdf):
    """Converts a multipart gdf to singlepart gdf """
    gdf['dissolvefield'] = 1
    gdf = gdf.dissolve(by='dissolvefield')
    gdf.reset_index(inplace=True)
    gdf = gdf[['geometry']] #remove all columns
         
    return gdf



def get_wkb_srid (gdf):
    """Returns SRID and WKB objects from gdf"""

    srid = gdf.crs.to_epsg()
    
    geom = gdf['geometry'].iloc[0]

    wkb_aoi = wkb.dumps(geom)
    
    # if geometry has Z values, flatten geometry
    if geom.has_z:
        wkb_aoi = wkb.dumps(geom, output_dimension=2)
        
    
    return wkb_aoi, srid



def read_input_spreadsheets (wksp_xls,region):
    """Returns input spreadhseets"""
    common_xls = os.path.join(wksp_xls, 'one_status_common_datasets.xlsx')
    region_xls = os.path.join(wksp_xls, 'one_status_{}_specific.xlsx'.format(region.lower()))
    
    df_stat_c = pd.read_excel(common_xls)
    df_stat_r = pd.read_excel(region_xls)
    
    df_stat = pd.concat([df_stat_c, df_stat_r])
    df_stat.dropna(how='all', inplace=True)
    
    df_stat = df_stat.reset_index(drop=True)
    
    return df_stat
    
    

def get_table_cols (item_index,df_stat):
    """Returns table and field names from the AST datasets spreadsheet"""
    df_stat_item = df_stat.loc[[item_index]]
    df_stat_item.fillna(value='nan',inplace=True)

    table = df_stat_item['Datasource'].iloc[0].strip()
    
    fields = []
    fields.append(str(df_stat_item['Fields_to_Summarize'].iloc[0].strip()))

    for f in range (2,7):
        for i in df_stat_item['Fields_to_Summarize' + str(f)].tolist():
            if i != 'nan':
                fields.append(str(i.strip()))

    col_lbl = df_stat_item['map_label_field'].iloc[0].strip()
    
    if col_lbl != 'nan' and col_lbl not in fields:
        fields.append(col_lbl)
    
    if table.startswith('WHSE') or table.startswith('REG'):       
        cols = ','.join('b.' + x for x in fields)

        # TEMPORARY FIX:  for empty column names in the COMMON AST input spreadsheet
        if cols == 'b.nan':
            cols = 'b.OBJECTID'
    else:
        cols = fields
        # TEMPORARY FIX:  for empty column names in the REGION AST input spreadsheet
        if cols[0] == 'nan':
            cols = []
    
    return table, cols, col_lbl

          

def get_def_query (item_index,df_stat):
    """Returns an ORacle SQL formatted def query (if any) from the AST datasets spreadsheet"""
    df_stat_item = df_stat.loc[[item_index]]
    df_stat_item.fillna(value='nan',inplace=True)

    def_query = df_stat_item['Definition_Query'].iloc[0].strip()

    def_query = def_query.strip()
    
    if def_query == 'nan':
        def_query = " "
        
    else:
       def_query = def_query.replace('"', '')
       def_query = re.sub(r'(\bAND\b)', r'\1 b.', def_query)
       def_query = re.sub(r'(\bOR\b)', r'\1 b.', def_query)
       
       if def_query[0] == "(":
           def_query = def_query.replace ("(", "(b.") 
           def_query = "(" + def_query + ")"
       else:
           def_query = "b." + def_query
    
       def_query = 'AND (' + def_query + ')'
    
    
    return def_query



def get_radius (item_index, df_stat):
    """Returns the buffer distance (if any) from the AST common datasets spreadsheet"""
    df_stat_item = df_stat.loc[[item_index]]
    df_stat_item.fillna(value=0,inplace=True)
    df_stat_item['Buffer_Distance'] = df_stat_item['Buffer_Distance'].astype(int)
    radius = df_stat_item['Buffer_Distance'].iloc[0]
    
    return radius


def apply_curve_fix(query, table_name, geom_col):
    """
    Apply geometry validation filter for specific problematic tables
    with Arc/Curve geometries that cannot be processed directly.
    
    Uses SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT to filter out only geometries
    with ORA-13347 arc errors, keeping all other valid geometries.
    """
    # List of tables that need geometry validation filter
    PROBLEMATIC_TABLES = [
        'WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW',
        'WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_SVW'
    ]
    
    # Only apply fix to problematic tables
    if table_name not in PROBLEMATIC_TABLES:
        return query
    
    print('.......applying SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT filter to exclude ORA-13347 arc errors')
    print('.......Note: Only geometries with arc coordinate errors (13347) will be excluded')
    
    # Add validation check to WHERE clause - only exclude 13347 errors
    validation_clause = f"SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(b.{geom_col}, 0.5) NOT LIKE '%13347%'\n  AND "
    
    # Insert the validation before the SDO_WITHIN_DISTANCE clause
    query = query.replace(
        'WHERE SDO_WITHIN_DISTANCE',
        f'WHERE {validation_clause}SDO_WITHIN_DISTANCE'
    )
    
    return query


def load_queries ():
    sql = {}

    sql ['aoi'] = """
                    SELECT SDO_UTIL.TO_WKTGEOMETRY(a.SHAPE) SHAPE
                    
                    FROM  WHSE_TANTALIS.TA_CROWN_TENURES_SVW a
                    
                    WHERE a.CROWN_LANDS_FILE = :file_nbr
                        AND a.DISPOSITION_TRANSACTION_SID = :disp_id
                        AND a.INTRID_SID = :parcel_id
                  """
                        
    sql ['geomCol'] = """
                    SELECT column_name GEOM_NAME
                    
                    FROM  ALL_SDO_GEOM_METADATA
                    
                    WHERE owner = :owner
                        AND table_name = :tab_name
                        
          
                    """    
                    
    sql ['srid'] = """
                    SELECT s.{geom_col}.sdo_srid SP_REF
                    FROM {tab} s
                    WHERE rownum = 1
                   """
                                 
    sql ['overlay_wkb'] = """
                    SELECT {cols},
                    
                           CASE WHEN SDO_GEOM.SDO_DISTANCE(b.{geom_col}, SDO_GEOMETRY(:wkb_aoi, :srid), 0.5) = 0 
                            THEN 'INTERSECT' 
                             ELSE 'Within ' || TO_CHAR({radius}) || ' m'
                              END AS RESULT,
                              
                           SDO_UTIL.TO_WKTGEOMETRY(b.{geom_col}) SHAPE
                    
                    FROM {tab} b
                    
                    WHERE SDO_WITHIN_DISTANCE (b.{geom_col}, 
                                               SDO_GEOMETRY(:wkb_aoi, :srid),'distance = {radius}') = 'TRUE'
                        {def_query}   
                                  """ 
    
    # Query for tables with SRID mismatch - transforms AOI to match table SRID
    sql ['overlay_wkb_transform'] = """
                    SELECT {cols},
                    
                           CASE WHEN SDO_GEOM.SDO_DISTANCE(b.{geom_col}, 
                                    SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t), 0.5) = 0 
                            THEN 'INTERSECT' 
                             ELSE 'Within ' || TO_CHAR({radius}) || ' m'
                              END AS RESULT,
                              
                           SDO_UTIL.TO_WKTGEOMETRY(SDO_CS.TRANSFORM(b.{geom_col}, :srid_t, :srid)) SHAPE
                    
                    FROM {tab} b
                    
                    WHERE SDO_WITHIN_DISTANCE (b.{geom_col}, 
                                SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t),'distance = {radius}') = 'TRUE'
                        {def_query}   
                    """
    return sql



def get_geom_colname (connection,cursor,table,geomQuery):
    """ Returns the geometry column of BCGW table name: can be either SHAPE or GEOMETRY"""
    el_list = table.split('.')

    bvars_geom = {'owner':el_list[0].strip(),
                  'tab_name':el_list[1].strip()}
    df_g = read_query(connection,cursor,geomQuery, bvars_geom)
    
    geom_col = df_g['GEOM_NAME'].iloc[0]

    return geom_col



def get_geom_srid (connection,cursor,table,geom_col,sridQuery):
    """ Returns the SRID of the BCGW table"""

    sridQuery = sridQuery.format(tab=table,geom_col=geom_col)
    df_s = read_query(connection,cursor,sridQuery,{})
    
    srid_t = df_s['SP_REF'].iloc[0]

    return srid_t



def make_status_map (gdf_aoi, gdf_intr, col_lbl, item, workspace):
    """ Generates HTML Interactive maps of AOI and intersection geodataframes"""
    
    m = folium.Map(tiles='openstreetmap')
    xmin,ymin,xmax,ymax = gdf_aoi.to_crs(4326)['geometry'].total_bounds
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])

    gdf_aoi.explore(
         m=m,
         tooltip= False,
         style_kwds=dict(fill= False, color="red", weight=3),
         name="AOI")

    gdf_intr.explore(
         m=m,
         column= col_lbl, 
         tooltip= col_lbl, 
         popup=True, 
         cmap="Dark2",  
         style_kwds=dict(color="gray"),
         name=item)
	
    folium.TileLayer('stamenterrain', control=True).add_to(m)
    folium.LayerControl().add_to(m)
    
    maps_dir = os.path.join(workspace,'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
        
    out_html = os.path.join(maps_dir, item +'.html')
    m.save(out_html)
 


def write_xlsx (results,df_stat,workspace):
    """Writes results to a spreadsheet"""
    df_res= df_stat[['Category', 'Featureclass_Name(valid characters only)']]   
    df_res.rename(columns={'Featureclass_Name(valid characters only)': 'item'}, inplace=True)
    df_res['List of conflicts'] = ""
    df_res['Map'] = ""
    
    # Create a list to hold expanded rows
    expanded_rows = []
    
    for index, row in df_res.iterrows():
        has_conflicts = False
        for k, v in results.items():
            if row['item'] == k and v.shape[0] > 0:
                has_conflicts = True
                v = v.drop('RESULT', axis=1)
                v['Result'] = v[v.columns].apply(lambda row: '; '.join(row.values.astype(str)), axis=1)
                
                # Create a separate row for each conflict
                for conflict in v['Result'].to_list():
                    expanded_rows.append({
                        'Category': row['Category'],
                        'item': row['item'],
                        'List of conflicts': str(conflict),
                        'Map': '=HYPERLINK("{}", "View Map")'.format(os.path.join(workspace,'maps',k+'.html'))
                    })
                break
        
        # If no conflicts found, add the original row with empty conflicts
        if not has_conflicts:
            expanded_rows.append({
                'Category': row['Category'],
                'item': row['item'],
                'List of conflicts': '',
                'Map': ''
            })
    
    # Create new dataframe from expanded rows
    df_res = pd.DataFrame(expanded_rows)

    filename = os.path.join(workspace, 'AST_lite_TAB3.xlsx')
    sheetname = 'Conflicts & Constraints'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')        
    df_res.to_excel(writer, sheet_name=sheetname, index=False, startrow=0 , startcol=0)
    
    workbook=writer.book
    worksheet = writer.sheets[sheetname]
    
    txt_format = workbook.add_format({'text_wrap': True})
    lnk_format = workbook.add_format({'underline': True, 'font_color': 'blue'})
    worksheet.set_column(0, 0, 30)
    worksheet.set_column(1, 1, 60)
    worksheet.set_column(2, 2, 80, txt_format)
    worksheet.set_column(3, 3, 20)
    
    worksheet.conditional_format('D2:D{}'.format (df_res.shape[0]+1), 
                                 {'type': 'cell',
                                  'criteria' : 'equal to', 
                                  'value' : '"View Map"',
                                  'format' : lnk_format})
    
    col_names = [{'header': col_name} for col_name in df_res.columns]
    worksheet.add_table(0, 0, df_res.shape[0]+1, df_res.shape[1]-1,{'columns': col_names})
    
    writer.close()


    
if __name__ == "__main__":
    """Executes the AST light process """
    start_t = timeit.default_timer() #start time
    
    #paths
    workspace = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
    wksp_xls = os.path.join(workspace, 'input_spreadsheets')
    aoi = os.path.join(workspace, 'test_data', 'aoi_test_3.shp')
    out_wksp = os.path.join(workspace, 'outputs')
    
    
    print ('Connecting to BCGW.')
    hostname = 'bcgw.bcgov/idwprod1.bcgov'
    bcgw_user = os.getenv('bcgw_user')
    #bcgw_user = 'XXXX'
    bcgw_pwd = os.getenv('bcgw_pwd')
    #bcgw_pwd = 'XXXX'
    connection, cursor = connect_to_Oracle (bcgw_user,bcgw_pwd,hostname)
    
    print ('\nLoading SQL queries')
    sql = load_queries ()
    
    
    print ('\nReading User inputs: AOI.')
    input_src = 'AOI' # Possible values are "TANTALIS" and AOI

    if input_src == 'AOI':
        print('....Reading the AOI file')
        gdf_aoi = esri_to_gdf (aoi)
    
        if gdf_aoi.shape[0] > 1:
            gdf_aoi =  multipart_to_singlepart(gdf_aoi)

       
    elif input_src == 'TANTALIS':
     
        #test tenure - big
        #fileNbr = '0327094'
        #dispID  =  146307
        #prclID  =  895143
        
        #test tenure - small
        fileNbr = '1413717'
        dispID  = 927132
        prclID  = 953951
     
        in_fileNbr = fileNbr
        in_dispID = dispID
        in_prclID = prclID
        print ('....input File Number: {}'.format(in_fileNbr))
        print ('....input Disposition ID: {}'.format(in_dispID))
        print ('....input Parcel ID: {}'.format(in_prclID))
        
        bvars_aoi = {'file_nbr': in_fileNbr,
                     'disp_id': in_dispID, 'parcel_id': in_prclID}
        
        print('....Querying TANTALIS for AOI geometry')
        df_aoi= read_query(connection,cursor,sql ['aoi'],bvars_aoi) 
        
        if df_aoi.shape[0] < 1:
            raise Exception('Parcel not in TANTALIS. Please check inputs!')
            
        else:
            print('....Converting TANTALIS result to GeoDataFrame')
            gdf_aoi = df_2_gdf (df_aoi, 3005)
    
                
    else:
        raise Exception('Possible input sources are TANTALIS and AOI!')
    
    # Convert AOI to WKB regardless of source (unified approach)
    print('....Extracting WKB and SRID from AOI')
    wkb_aoi, srid = get_wkb_srid (gdf_aoi)
    
    print ('\nReading the AST datasets spreadsheet.')
    region = 'west_coast' #**************USER INPUT: REGION*************
    print ('....Region is {}'.format (region))
    df_stat = read_input_spreadsheets (wksp_xls,region)
    
    
    
    print ('\nRunning the analysis.')
    results = {} # this dictionnary will hold the overlay results
    
    item_count = df_stat.shape[0]
    counter = 1
    for index, row in df_stat.iterrows():
        item = row['Featureclass_Name(valid characters only)']
        item_index = index
        
        print ('\n****working on item {} of {}: {}***'.format(counter,item_count,item))
        
        print ('.....getting table and column names')
        table, cols, col_lbl = get_table_cols (item_index,df_stat)
        
        print ('.....getting definition query (if any)')
        def_query = get_def_query (item_index,df_stat)
    
        print ('.....getting buffer distance (if any)')
        radius = get_radius (item_index, df_stat)  
         
        print ('.....running Overlay Analysis.')
        
        if table.startswith('WHSE') or table.startswith('REG'): 
            geomQuery = sql ['geomCol']
            sridQuery = sql ['srid']
            geom_col = get_geom_colname (connection,cursor,table,geomQuery)
            
            # Check if SRID mismatch exists
            srid_t = get_geom_srid (connection,cursor,table,geom_col,sridQuery) 
            srid_mismatch = (srid_t == 1000003005)
            
            if srid_mismatch:
                print(f'.......SRID mismatch detected (table SRID: {srid_t}), using transform query')
                query = sql ['overlay_wkb_transform'].format(
                    cols=cols, tab=table, radius=radius,
                    geom_col=geom_col, def_query=def_query)
                cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
                bvars_intr = {'wkb_aoi':wkb_aoi, 'srid':int(srid), 'srid_t':int(srid_t)}
            else:
                # Use standard WKB query for matching SRIDs
                query = sql ['overlay_wkb'].format(
                    cols=cols, tab=table, radius=radius,
                    geom_col=geom_col, def_query=def_query)
                cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
                bvars_intr = {'wkb_aoi':wkb_aoi, 'srid':int(srid)}
            
            # Apply RECTIFY fix for problematic tables
            query = apply_curve_fix(query, table, geom_col)
            
            df_all= read_query(connection,cursor,query,bvars_intr)
            
                
        else:
            try:
                gdf_trg = esri_to_gdf (table)
                
                if not gdf_trg.crs.to_epsg() == 3005:
                    gdf_trg = gdf_trg.to_crs({'init': 'epsg:3005'})
                    
                gdf_intr = gpd.overlay(gdf_aoi, gdf_trg, how='intersection')
                
                
                # TEMPORARY FIX:  for Empty/Wrong column names in the REGION AST input spreadsheet
                gdf_cols = [col for col in gdf_trg.columns]  
                diffs = list(set(cols).difference(gdf_cols))
                for diff in diffs:
                    cols.remove(diff)
                if len(cols) ==0:
                    cols.append(gdf_trg.columns[0])
                 
                df_intr = pd.DataFrame(gdf_intr)
                df_intr ['RESULT'] = 'INTERSECT'
                
                if radius > 0:
                    aoi_buf = gdf_aoi.buffer(radius)
                    gdf_aoi_buf = gpd.GeoDataFrame(gpd.GeoSeries(aoi_buf))
                    gdf_aoi_buf = gdf_aoi_buf.rename(columns={0:'geometry'}).set_geometry('geometry')
                    gdf_aoi_buf_ext = gpd.overlay(gdf_aoi, gdf_aoi_buf, how='symmetric_difference')  
                    gdf_buf= gpd.overlay(gdf_aoi_buf_ext, gdf_trg, how='intersection')
                    
                    df_buf = pd.DataFrame(gdf_buf)
                    df_buf ['RESULT'] = 'WITHIN {} m'.format(str(radius))   
                    
                    df_all =  pd.concat([df_intr, df_buf])
                    
                else:
                    df_all = df_intr
                
                df_all.rename(columns={'geometry':'SHAPE'},inplace=True)
                
            except:
                print ('.......ERROR: the Source Dataset does NOT exist!')
                df_all = pd.DataFrame([])
        
        
        if isinstance(cols, str) == True:
            l = cols.split(",")
            cols = [x[2:] for x in l]
    
        cols.append('RESULT')
        
        df_all_res = df_all[cols]  
        
        
        ov_nbr = df_all_res.shape[0]
        print ('.....number of overlaps: {}'.format(ov_nbr))
        
        # add the dataframe to the resuls dictionnary
        results[item] =  df_all_res
        '''
        if ov_nbr > 0:
            print ('.....generating a map.')
            gdf_intr = df_2_gdf (df_all, 3005)
            
            # FIX FOR MISSING LABEL COLUMN NAME
            if col_lbl == 'nan': 
                col_lbl = cols[0]
                gdf_intr [col_lbl] = gdf_intr [col_lbl].astype(str)
            
            # datetime columns are causing errors when plotting in Folium. Converting them to str
            for col in gdf_intr.columns:
                if gdf_intr[col].dtype == 'datetime64[ns]':
                    gdf_intr[col] = gdf_intr[col].astype(str)
            
            gdf_intr[col_lbl] = gdf_intr[col_lbl].astype(str) 
            
            make_status_map (gdf_aoi, gdf_intr, col_lbl, item, out_wksp)
        '''
        
        counter += 1
    
    print ('\nWriting Results to spreadsheet')
    write_xlsx (results,df_stat,out_wksp)
    
    finish_t = timeit.default_timer() #finish time
    t_sec = round(finish_t-start_t)
    mins = int (t_sec/60)
    secs = int (t_sec%60)
    print ('\nProcessing Completed in {} minutes and {} seconds'.format (mins,secs))