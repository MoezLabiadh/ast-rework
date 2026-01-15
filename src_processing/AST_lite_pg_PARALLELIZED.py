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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import oracledb
import psycopg2
import pandas as pd
import folium
import geopandas as gpd
from shapely import wkb


# ============================================================================
# THREAD-SAFE UTILITIES
# ============================================================================

class ThreadSafeResources:
    """Thread-safe resource management for parallel processing."""
    
    def __init__(self):
        self.results_lock = Lock()
        self.print_lock = Lock()
    
    def safe_print(self, message: str) -> None:
        """Thread-safe print function."""
        with self.print_lock:
            print(message)


# Global thread-safe resources
_thread_resources = ThreadSafeResources()


# ============================================================================
# DATABASE CONNECTION UTILITIES
# ============================================================================

def connect_to_oracle(username: str, password: str, hostname: str) -> Tuple[Any, Any]:
    """
    Create Oracle database connection.
    
    Args:
        username: Oracle username
        password: Oracle password
        hostname: Oracle hostname/DSN
    
    Returns:
        Tuple of (connection, cursor)
    """
    try:
        connection = oracledb.connect(user=username, password=password, dsn=hostname)
        cursor = connection.cursor()
        return connection, cursor
    except Exception as e:
        raise Exception(f'Oracle connection failed! Error: {e}')


def connect_to_postgis(
    host: str, 
    database: str, 
    user: str, 
    password: str, 
    port: int = 5432
) -> Tuple[Any, Any]:
    """
    Create PostGIS database connection.
    
    Args:
        host: PostgreSQL host
        database: Database name
        user: PostgreSQL user
        password: PostgreSQL password
        port: PostgreSQL port
    
    Returns:
        Tuple of (connection, cursor)
    """
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
        raise Exception(f'PostGIS connection failed! Error: {e}')


def read_query(connection: Any, cursor: Any, query: str, bind_vars: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.
    
    Args:
        connection: Database connection
        cursor: Database cursor
        query: SQL query string
        bind_vars: Dictionary of bind variables
    
    Returns:
        DataFrame containing query results
    """
    cursor.execute(query, bind_vars)
    names = [x[0] for x in cursor.description]
    rows = cursor.fetchall()
    
    # Convert any LOB objects to strings immediately
    processed_rows = []
    for row in rows:
        processed_row = []
        for val in row:
            if hasattr(val, 'read'):  # LOB objects have a read method
                try:
                    processed_row.append(str(val))
                except:
                    processed_row.append(val)
            else:
                processed_row.append(val)
        processed_rows.append(tuple(processed_row))
    
    return pd.DataFrame(processed_rows, columns=names)


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

class GeometryProcessor:
    """Handles geometry operations and conversions."""
    
    @staticmethod
    def esri_to_gdf(aoi_path: str) -> gpd.GeoDataFrame:
        """Convert ESRI format vector to GeoDataFrame."""
        if '.shp' in aoi_path:
            return gpd.read_file(aoi_path)
        elif '.gdb' in aoi_path:
            parts = aoi_path.split('.gdb')
            gdb = parts[0] + '.gdb'
            fc = os.path.basename(aoi_path)
            return gpd.read_file(filename=gdb, layer=fc)
        else:
            raise ValueError('Format not recognized. Please provide a shp or featureclass (gdb)!')
    
    @staticmethod
    def df_to_gdf(df: pd.DataFrame, crs: int) -> gpd.GeoDataFrame:
        """Convert DataFrame with geometry column to GeoDataFrame."""
        # Determine geometry column name
        if 'SHAPE' in df.columns:
            shape_col = 'SHAPE'
        elif 'shape' in df.columns:
            shape_col = 'shape'
        else:
            raise ValueError("No geometry column found. Expected 'SHAPE' or 'shape'")
        
        # Handle both string and LOB data types
        if df[shape_col].dtype == 'object':
            first_val = df[shape_col].iloc[0]
            if isinstance(first_val, str):
                shape_series = df[shape_col]
            else:
                shape_series = df[shape_col].apply(lambda x: str(x) if x is not None else None)
        else:
            shape_series = df[shape_col].astype(str)
        
        # Process geometries
        processed_wkts = [
            GeometryProcessor._process_geometry(wkt) 
            for wkt in shape_series
        ]
        
        # Create clean DataFrame
        df_clean = df.drop(columns=[shape_col]).copy()
        df_clean['geometry'] = gpd.GeoSeries.from_wkt(processed_wkts, crs=f"EPSG:{crs}")
        
        return gpd.GeoDataFrame(df_clean, geometry='geometry', crs=f"EPSG:{crs}")
    
    @staticmethod
    def _process_geometry(wkt_str: str) -> Optional[str]:
        """Process geometry string, linearizing curves if needed."""
        if wkt_str is None or not isinstance(wkt_str, str):
            return None
        
        curve_types = ['CURVE', 'CIRCULARSTRING', 'COMPOUNDCURVE']
        if any(curve_type in wkt_str.upper() for curve_type in curve_types):
            return GeometryProcessor._linearize_geometry(wkt_str)
        
        return wkt_str
    
    @staticmethod
    def _linearize_geometry(wkt_str: str) -> Optional[str]:
        """Convert curve geometries to linear approximations."""
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
    
    @staticmethod
    def multipart_to_singlepart(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert multipart GeoDataFrame to singlepart."""
        gdf['dissolvefield'] = 1
        gdf = gdf.dissolve(by='dissolvefield')
        gdf.reset_index(inplace=True)
        return gdf[['geometry']]
    
    @staticmethod
    def get_wkb_srid(gdf: gpd.GeoDataFrame) -> Tuple[bytes, int]:
        """Extract WKB and SRID from GeoDataFrame."""
        srid = gdf.crs.to_epsg()
        geom = gdf['geometry'].iloc[0]
        
        if geom.has_z:
            wkb_aoi = wkb.dumps(geom, output_dimension=2)
        else:
            wkb_aoi = wkb.dumps(geom)
        
        return wkb_aoi, srid
    
    @staticmethod
    def simplify_geometries(
        gdf: gpd.GeoDataFrame, 
        tolerance: float = 10, 
        preserve_topology: bool = True
    ) -> gpd.GeoDataFrame:
        """Simplify geometries for web map display."""
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=tolerance, 
            preserve_topology=preserve_topology
        )
        return gdf


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

class DatasetConfig:
    """Handles reading and parsing dataset configuration spreadsheets."""
    
    @staticmethod
    def read_spreadsheets(workspace_xls: str, region: str) -> pd.DataFrame:
        """Read and combine common and region-specific dataset spreadsheets."""
        common_xls = os.path.join(workspace_xls, 'one_status_common_datasets.xlsx')
        region_xls = os.path.join(workspace_xls, f'one_status_{region.lower()}_specific.xlsx')
        
        df_common = pd.read_excel(common_xls)
        df_region = pd.read_excel(region_xls)
        
        df_combined = pd.concat([df_common, df_region])
        df_combined.dropna(how='all', inplace=True)
        df_combined.reset_index(drop=True, inplace=True)
        
        return df_combined
    
    @staticmethod
    def get_table_columns(item_index: int, df_stat: pd.DataFrame) -> Tuple[str, str, str]:
        """Extract table name and column information from config."""
        df_item = df_stat.loc[[item_index]].fillna('nan')
        
        table = df_item['Datasource'].iloc[0].strip()
        
        fields = []
        first_field = str(df_item['Fields_to_Summarize'].iloc[0].strip())
        if first_field != 'nan':
            fields.append(first_field)
        
        for i in range(2, 7):
            col_name = f'Fields_to_Summarize{i}'
            if col_name in df_item.columns:
                for field in df_item[col_name].tolist():
                    if field != 'nan':
                        fields.append(str(field).strip())
        
        label_col = df_item['map_label_field'].iloc[0].strip()
        if label_col != 'nan' and label_col not in fields:
            fields.append(label_col)
        
        cols_csv = ','.join(fields) if fields else ''
        
        return table, cols_csv, label_col
    
    @staticmethod
    def get_definition_query(item_index: int, df_stat: pd.DataFrame, for_postgis: bool = False) -> str:
        """Extract and format definition query from config."""
        df_item = df_stat.loc[[item_index]].fillna('nan')
        def_query = df_item['Definition_Query'].iloc[0].strip()
        
        if def_query == 'nan' or not def_query:
            return ""
        
        def_query = def_query.replace('"', '')
        
        if for_postgis:
            def_query = def_query.replace('%', '%%')
        
        return f'AND ({def_query})'
    
    @staticmethod
    def get_buffer_distance(item_index: int, df_stat: pd.DataFrame) -> int:
        """Extract buffer distance from config."""
        df_item = df_stat.loc[[item_index]].fillna(0)
        df_item['Buffer_Distance'] = df_item['Buffer_Distance'].astype(int)
        return df_item['Buffer_Distance'].iloc[0]


# ============================================================================
# SQL QUERY TEMPLATES
# ============================================================================

def load_sql_queries() -> Dict[str, str]:
    """Load all SQL query templates."""
    return {
        'aoi': """
            SELECT SDO_UTIL.TO_WKTGEOMETRY(a.SHAPE) SHAPE
            FROM WHSE_TANTALIS.TA_CROWN_TENURES_SVW a
            WHERE a.CROWN_LANDS_FILE = :file_nbr
                AND a.DISPOSITION_TRANSACTION_SID = :disp_id
                AND a.INTRID_SID = :parcel_id
        """,
        
        'geomCol': """
            SELECT column_name GEOM_NAME
            FROM ALL_SDO_GEOM_METADATA
            WHERE owner = :owner
                AND table_name = :tab_name
        """,
        
        'srid': """
            SELECT s.{geom_col}.sdo_srid SP_REF
            FROM {tab} s
            WHERE rownum = 1
        """,
        
        # Oracle overlay query with two-stage filtering optimization:
        # Uses SDO_RELATE for direct overlaps (radius=0)
        # Uses SDO_WITHIN_DISTANCE for buffer/proximity checks (radius>0)
        
        # Query for radius = 0 (direct overlap only)
        'oracle_overlay_zero_radius': """
            SELECT {cols},
                   'INTERSECT' AS RESULT,
                   SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE
            FROM {tab}
            WHERE SDO_FILTER({geom_col}, 
                             SDO_GEOMETRY(:wkb_aoi, :srid),
                             'querytype=WINDOW') = 'TRUE'
              AND SDO_RELATE({geom_col}, 
                             SDO_GEOMETRY(:wkb_aoi, :srid),
                             'mask=ANYINTERACT') = 'TRUE'
                {def_query}
        """,
        
        # Query for radius > 0 (buffer/distance check)
        'oracle_overlay_with_radius': """
            SELECT {cols},
                   CASE WHEN SDO_GEOM.SDO_DISTANCE({geom_col}, SDO_GEOMETRY(:wkb_aoi, :srid), 0.5) = 0 
                    THEN 'INTERSECT' 
                     ELSE 'Within ' || TO_CHAR({radius}) || ' m'
                      END AS RESULT,
                   SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE
            FROM {tab}
            WHERE SDO_FILTER({geom_col}, 
                             SDO_GEOM.SDO_BUFFER(SDO_GEOMETRY(:wkb_aoi, :srid), {radius}, 0.5),
                             'querytype=WINDOW') = 'TRUE'
              AND SDO_WITHIN_DISTANCE ({geom_col}, 
                                       SDO_GEOMETRY(:wkb_aoi, :srid),'distance = {radius}') = 'TRUE'
                {def_query}
        """,
    }


# ============================================================================
# ORACLE DATABASE UTILITIES
# ============================================================================

class OracleUtils:
    """Utilities for working with Oracle/BCGW datasets."""
    
    PROBLEMATIC_TABLES = [
        'WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW',
        'WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_SVW'
    ]
    
    @staticmethod
    def get_geometry_column(connection: Any, cursor: Any, table: str, geom_query: str) -> str:
        """Get geometry column name for Oracle table."""
        parts = table.split('.')
        bind_vars = {'owner': parts[0].strip(), 'tab_name': parts[1].strip()}
        df = read_query(connection, cursor, geom_query, bind_vars)
        return df['GEOM_NAME'].iloc[0]
    
    @staticmethod
    def get_srid(connection: Any, cursor: Any, table: str, geom_col: str, srid_query: str) -> Optional[int]:
        """Get SRID for Oracle table."""
        try:
            query = srid_query.format(tab=table, geom_col=geom_col)
            df = read_query(connection, cursor, query, {})
            
            if df.empty or df.shape[0] == 0:
                return None
            
            return df['SP_REF'].iloc[0]
        except:
            return None
    
    @staticmethod
    def get_columns(connection: Any, cursor: Any, table: str) -> List[str]:
        """Retrieve list of available columns for Oracle table."""
        try:
            query = """
                SELECT column_name 
                FROM all_tab_columns 
                WHERE owner = :owner 
                    AND table_name = :tab_name
            """
            parts = table.split('.')
            bind_vars = {'owner': parts[0].strip(), 'tab_name': parts[1].strip()}
            df = read_query(connection, cursor, query, bind_vars)
            
            return df['COLUMN_NAME'].tolist() if not df.empty else []
        except:
            return []
    
    @staticmethod
    def apply_geometry_fix(query: str, table: str, geom_col: str) -> str:
        """Apply geometry densification and rectification for problematic Oracle tables.
        
        This fixes:
        - Curved geometries (CURVEPOLYGON, COMPOUNDCURVE) that GeoPandas can't handle
        - Invalid geometries (ring orientation, self-intersections, etc.)
        
        Uses SDO_GEOM.SDO_ARC_DENSIFY to convert curves to line segments
        and SDO_UTIL.RECTIFY_GEOMETRY to fix geometry errors.
        
        Args:
            query: SQL query string
            table: Table name
            geom_col: Geometry column name
        
        Returns:
            Modified query with geometry fix applied
        """
        if table not in OracleUtils.PROBLEMATIC_TABLES:
            return query
        
        # Replace the geometry output with densified and rectified version
        # Original: SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE
        # Fixed: Densify curves (arc_tolerance=0.5m), then rectify any remaining issues
        
        original_wkt = f'SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE'
        fixed_wkt = (
            f'SDO_UTIL.TO_WKTGEOMETRY('
            f'SDO_UTIL.RECTIFY_GEOMETRY('
            f'SDO_GEOM.SDO_ARC_DENSIFY({geom_col}, 0.005, \'arc_tolerance=0.5\'), '
            f'0.005)) SHAPE'
        )
        
        return query.replace(original_wkt, fixed_wkt)
    
    @staticmethod
    def apply_coordinate_transform(query: str, geom_col: str, srid_t: int) -> str:
        """Apply coordinate transformation when SRIDs don't match."""
        query = query.replace(
            'SDO_GEOMETRY(:wkb_aoi, :srid), 0.5)',
            'SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t), 0.5)'
        )
        
        query = query.replace(
            'SDO_GEOMETRY(:wkb_aoi, :srid),',
            'SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t),'
        )
        
        query = query.replace(
            f'SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE',
            f'SDO_UTIL.TO_WKTGEOMETRY(SDO_CS.TRANSFORM({geom_col}, :srid_t, :srid)) SHAPE'
        )
        
        return query


# ============================================================================
# POSTGIS DATABASE UTILITIES
# ============================================================================

class PostGISUtils:
    """Utilities for working with PostGIS datasets."""
    
    @staticmethod
    def get_table_name_from_datasource(datasource: str) -> str:
        """Extract and clean table name from datasource path."""
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
    
    @staticmethod
    def table_exists(cursor: Any, schema: str, table: str) -> bool:
        """Check if PostGIS table exists."""
        try:
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s 
                        AND table_name = %s
                )
            """
            cursor.execute(query, (schema, table))
            return cursor.fetchone()[0]
        except Exception as e:
            try:
                cursor.connection.rollback()
            except:
                pass
            return False
    
    @staticmethod
    def get_columns(cursor: Any, schema: str, table: str) -> List[str]:
        """Retrieve list of available columns for PostGIS table."""
        try:
            query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = %s 
                    AND table_name = %s
                    AND column_name != 'geometry'
            """
            cursor.execute(query, (schema, table))
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            try:
                cursor.connection.rollback()
            except:
                pass
            return []


# ============================================================================
# COLUMN VALIDATION
# ============================================================================

class ColumnValidator:
    """Validates and normalizes column names for database queries."""
    
    @staticmethod
    def convert_to_uppercase(cols_str: str) -> str:
        """Convert comma-separated column names to uppercase for Oracle."""
        if not cols_str:
            return ''
        cols_list = [c.strip().upper() for c in cols_str.split(',')]
        return ','.join(cols_list)
    
    @staticmethod
    def convert_to_lowercase(cols_str: str) -> str:
        """Convert comma-separated column names to lowercase for PostGIS."""
        if not cols_str:
            return ''
        cols_list = [c.strip().lower() for c in cols_str.split(',')]
        return ','.join(cols_list)
    
    @staticmethod
    def validate_columns(
        cols: str, 
        available_cols: List[str], 
        item: str, 
        table: str, 
        is_postgis: bool = False
    ) -> Tuple[str, List[str]]:
        """Validate that requested columns exist in dataset."""
        missing_cols = []
        
        requested = [c.strip() for c in cols.split(',')] if cols else []
        
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


# ============================================================================
# MAP GENERATION
# ============================================================================

class MapGenerator:
    """Generates interactive HTML maps."""
    
    @staticmethod
    def create_status_map(
        gdf_aoi: gpd.GeoDataFrame,
        gdf_intersect: gpd.GeoDataFrame,
        label_col: str,
        item_name: str,
        workspace: str
    ) -> None:
        """Generate interactive HTML map showing AOI and intersecting features."""
        m = folium.Map(tiles='openstreetmap')
        xmin, ymin, xmax, ymax = gdf_aoi.to_crs(4326)['geometry'].total_bounds
        m.fit_bounds([[ymin, xmin], [ymax, xmax]])
        
        gdf_aoi.explore(
            m=m,
            tooltip=False,
            style_kwds=dict(fill=False, color="red", weight=3),
            name="AOI"
        )
        
        gdf_intersect.explore(
            m=m,
            column=label_col,
            tooltip=label_col,
            popup=True,
            cmap="Dark2",
            style_kwds=dict(color="gray"),
            name=item_name
        )
        
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        maps_dir = os.path.join(workspace, 'maps')
        os.makedirs(maps_dir, exist_ok=True)
        
        out_html = os.path.join(maps_dir, f'{item_name}.html')
        m.save(out_html)


# ============================================================================
# EXCEL REPORT GENERATION
# ============================================================================

class ExcelReportWriter:
    """Writes analysis results to Excel format."""
    
    @staticmethod
    def write_report(results: Dict[str, pd.DataFrame], df_stat: pd.DataFrame, workspace: str) -> None:
        """Write results to Excel spreadsheet."""
        df_res = df_stat[['Category', 'Featureclass_Name(valid characters only)']].copy()
        df_res.rename(columns={'Featureclass_Name(valid characters only)': 'item'}, inplace=True)
        df_res['List of conflicts'] = ""
        df_res['Map'] = ""
        
        expanded_rows = []
        
        for _, row in df_res.iterrows():
            has_conflicts = False
            
            for item_name, result_df in results.items():
                if row['item'] == item_name and result_df.shape[0] > 0:
                    has_conflicts = True
                    
                    result_col = None
                    if 'RESULT' in result_df.columns:
                        result_col = 'RESULT'
                    elif 'result' in result_df.columns:
                        result_col = 'result'
                    
                    if result_col:
                        result_df = result_df.drop(result_col, axis=1)
                    
                    result_df['Result'] = result_df[result_df.columns].apply(
                        lambda r: '; '.join(r.values.astype(str)), 
                        axis=1
                    )
                    
                    for conflict in result_df['Result'].to_list():
                        map_path = os.path.join(workspace, 'maps', f'{item_name}.html')
                        expanded_rows.append({
                            'Category': row['Category'],
                            'item': row['item'],
                            'List of conflicts': str(conflict),
                            'Map': f'=HYPERLINK("{map_path}", "View Map")'
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
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, sheet_name=sheetname, index=False, startrow=0, startcol=0)
            
            workbook = writer.book
            worksheet = writer.sheets[sheetname]
            
            txt_format = workbook.add_format({'text_wrap': True})
            lnk_format = workbook.add_format({'underline': True, 'font_color': 'blue'})
            
            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, 1, 60)
            worksheet.set_column(2, 2, 80, txt_format)
            worksheet.set_column(3, 3, 20)
            
            worksheet.conditional_format(
                f'D2:D{df_res.shape[0] + 1}',
                {
                    'type': 'cell',
                    'criteria': 'equal to',
                    'value': '"View Map"',
                    'format': lnk_format
                }
            )
            
            col_names = [{'header': col_name} for col_name in df_res.columns]
            worksheet.add_table(
                0, 0, df_res.shape[0] + 1, df_res.shape[1] - 1,
                {'columns': col_names}
            )


# ============================================================================
# DATASET PROCESSOR (PARALLELIZED)
# ============================================================================

class DatasetProcessor:
    """Processes individual datasets - designed for parallel execution."""
    
    def __init__(self, db_config: Dict[str, Any], sql_queries: Dict[str, str]):
        self.db_config = db_config
        self.sql = sql_queries
    
    def process_dataset(
        self,
        index: int,
        row: pd.Series,
        df_stat: pd.DataFrame,
        wkb_aoi: bytes,
        srid: int,
        region: str,
        gdf_aoi: gpd.GeoDataFrame
    ) -> Dict[str, Any]:
        """
        Process a single dataset with its own database connections.
        
        Returns:
            Dictionary containing processing results
        """
        item = row['Featureclass_Name(valid characters only)']
        
        connection = None
        cursor = None
        pg_connection = None
        pg_cursor = None
        
        try:
            _thread_resources.safe_print(f'🔄 Processing: {item}')
            
            # Get dataset parameters
            table, cols, col_lbl = DatasetConfig.get_table_columns(index, df_stat)
            def_query = DatasetConfig.get_definition_query(index, df_stat, for_postgis=False)
            radius = DatasetConfig.get_buffer_distance(index, df_stat)
            
            # Determine if Oracle or PostGIS
            is_oracle = table.startswith('WHSE') or table.startswith('REG')
            
            if is_oracle:
                connection, cursor = self._get_oracle_connection()
                result = self._process_oracle_dataset(
                    item, table, cols, col_lbl, def_query, radius, wkb_aoi, srid,
                    connection, cursor
                )
            else:
                pg_connection, pg_cursor = self._get_postgis_connection()
                result = self._process_postgis_dataset(
                    item, table, cols, col_lbl, def_query, radius, wkb_aoi, srid,
                    region, pg_connection, pg_cursor
                )
            
            return result
            
        except Exception as e:
            _thread_resources.safe_print(f'❌ {item}: ERROR - {str(e)[:100]}')
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
            if cursor:
                cursor.close()
            if connection:
                connection.close()
            if pg_cursor:
                pg_cursor.close()
            if pg_connection:
                pg_connection.close()
    
    def _get_oracle_connection(self) -> Tuple[Any, Any]:
        """Create Oracle connection using stored config."""
        return connect_to_oracle(
            self.db_config['oracle_user'],
            self.db_config['oracle_pwd'],
            self.db_config['oracle_host']
        )
    
    def _get_postgis_connection(self) -> Tuple[Any, Any]:
        """Create PostGIS connection using stored config."""
        return connect_to_postgis(
            self.db_config['pg_host'],
            self.db_config['pg_database'],
            self.db_config['pg_user'],
            self.db_config['pg_pwd'],
            self.db_config['pg_port']
        )
    
    def _process_oracle_dataset(
        self,
        item: str,
        table: str,
        cols: str,
        col_lbl: str,
        def_query: str,
        radius: int,
        wkb_aoi: bytes,
        srid: int,
        connection: Any,
        cursor: Any
    ) -> Dict[str, Any]:
        """Process Oracle/BCGW dataset."""
        # Get geometry column
        geom_col = OracleUtils.get_geometry_column(connection, cursor, table, self.sql['geomCol'])
        
        # Get SRID
        srid_t = OracleUtils.get_srid(connection, cursor, table, geom_col, self.sql['srid'])
        if srid_t is None:
            _thread_resources.safe_print(f'⚠️  {item}: Table is empty - SKIPPED')
            return self._create_error_result(item, 'Empty table')
        
        # Validate columns
        available_cols = OracleUtils.get_columns(connection, cursor, table)
        if not available_cols:
            _thread_resources.safe_print(f'⚠️  {item}: Could not retrieve columns - SKIPPED')
            return self._create_error_result(item, 'Could not retrieve table columns')
        
        cols_upper = ColumnValidator.convert_to_uppercase(cols)
        validated_cols, missing_cols = ColumnValidator.validate_columns(
            cols_upper, available_cols, item, table, is_postgis=False
        )
        
        if missing_cols:
            _thread_resources.safe_print(
                f'⚠️  {item}: Missing columns: {", ".join(missing_cols[:3])}'
                f'{"..." if len(missing_cols) > 3 else ""}'
            )
        
        # Clean def_query
        clean_def_query = def_query.strip()
        if clean_def_query and not clean_def_query.upper().startswith('AND'):
            clean_def_query = 'AND ' + clean_def_query
        
        # Build query
        query = self.sql['oracle_overlay'].format(
            cols=validated_cols,
            tab=table,
            radius=radius,
            geom_col=geom_col,
            def_query=clean_def_query
        )
        
        # Handle coordinate transformation
        srid_mismatch = (srid_t == 1000003005)
        if srid_mismatch:
            query = OracleUtils.apply_coordinate_transform(query, geom_col, srid_t)
            bind_vars = {'wkb_aoi': wkb_aoi, 'srid': int(srid), 'srid_t': int(srid_t)}
        else:
            bind_vars = {'wkb_aoi': wkb_aoi, 'srid': int(srid)}
        
        cursor.setinputsizes(wkb_aoi=oracledb.DB_TYPE_BLOB)
        query = OracleUtils.apply_geometry_fix(query, table, geom_col)
        
        # Execute query
        df_all = read_query(connection, cursor, query, bind_vars)
        
        # Process results
        col_lbl_upper = col_lbl.upper() if col_lbl != 'nan' else 'nan'
        return self._process_query_results(
            item, df_all, validated_cols, 'RESULT', col_lbl_upper
        )
    
    def _process_postgis_dataset(
        self,
        item: str,
        table: str,
        cols: str,
        col_lbl: str,
        def_query: str,
        radius: int,
        wkb_aoi: bytes,
        srid: int,
        region: str,
        pg_connection: Any,
        pg_cursor: Any
    ) -> Dict[str, Any]:
        """Process PostGIS dataset."""
        table_name = PostGISUtils.get_table_name_from_datasource(table)
        schema = region.lower()
        
        # Check if table exists
        if not PostGISUtils.table_exists(pg_cursor, schema, table_name):
            _thread_resources.safe_print(f'⚠️  {item}: Table not found in PostGIS - SKIPPED')
            return self._create_error_result(item, f'Table not found: {schema}.{table_name}')
        
        # Validate columns
        available_cols = PostGISUtils.get_columns(pg_cursor, schema, table_name)
        if not available_cols:
            _thread_resources.safe_print(f'⚠️  {item}: No columns found - SKIPPED')
            return self._create_error_result(item, f'No columns found in {schema}.{table_name}')
        
        cols_lower = ColumnValidator.convert_to_lowercase(cols)
        validated_cols, missing_cols = ColumnValidator.validate_columns(
            cols_lower, available_cols, item, table_name, is_postgis=True
        )
        
        if missing_cols:
            _thread_resources.safe_print(
                f'⚠️  {item}: Missing columns: {", ".join(missing_cols[:3])}'
                f'{"..." if len(missing_cols) > 3 else ""}'
            )
        
        # Handle definition query
        df_temp = pd.DataFrame([{'Definition_Query': def_query if def_query.strip() else 'nan'}])
        pg_def_query_raw = DatasetConfig.get_definition_query(0, df_temp, for_postgis=True)
        pg_def_query = pg_def_query_raw.replace('AND (', '(').replace('AND(', '(') if pg_def_query_raw else ''
        
        if pg_def_query and pg_def_query.strip():
            pg_def_query = 'AND ' + pg_def_query
        
        # Build query
        query = self.sql['postgis_overlay'].format(
            cols=validated_cols,
            schema=schema,
            table=table_name,
            def_query=pg_def_query
        )
        
        # Execute query
        try:
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
            
        except psycopg2.Error as e:
            try:
                pg_cursor.connection.rollback()
            except:
                pass
            raise Exception(f'PostGIS query error: {str(e)}')
        
        # Process results
        col_lbl_lower = col_lbl.lower() if col_lbl != 'nan' else 'nan'
        return self._process_query_results(
            item, df_all, validated_cols, 'result', col_lbl_lower
        )
    
    def _process_query_results(
        self,
        item: str,
        df_all: pd.DataFrame,
        validated_cols: str,
        result_col: str,
        col_lbl: str
    ) -> Dict[str, Any]:
        """Process query results and prepare return dictionary."""
        cols_list = [c.strip() for c in validated_cols.split(",")]
        
        if result_col in df_all.columns:
            cols_list.append(result_col)
        
        available_cols = [col for col in cols_list if col in df_all.columns]
        
        if not available_cols:
            _thread_resources.safe_print(f'⚠️  {item}: No valid columns in results - SKIPPED')
            return self._create_success_result(item)
        
        df_result = df_all[available_cols]
        ov_nbr = df_result.shape[0]
        
        if ov_nbr > 0:
            _thread_resources.safe_print(f'✅ {item}: {ov_nbr} overlap{"s" if ov_nbr != 1 else ""} found')
            
            # Create GeoDataFrame
            gdf_intr = GeometryProcessor.df_to_gdf(df_all, 3005)
            
            # Determine label column
            col_lbl_final = col_lbl if col_lbl != 'nan' else cols_list[0]
            
            if col_lbl_final in gdf_intr.columns:
                gdf_intr[col_lbl_final] = gdf_intr[col_lbl_final].astype(str)
            else:
                col_lbl_final = cols_list[0]
                if col_lbl_final in gdf_intr.columns:
                    gdf_intr[col_lbl_final] = gdf_intr[col_lbl_final].astype(str)
            
            # Convert datetime columns
            for col in gdf_intr.columns:
                if gdf_intr[col].dtype == 'datetime64[ns]':
                    gdf_intr[col] = gdf_intr[col].astype(str)
            
            return {
                'item': item,
                'df_result': df_result,
                'gdf_intr': gdf_intr,
                'col_lbl': col_lbl_final,
                'overlaps': ov_nbr,
                'success': True,
                'error': None
            }
        else:
            _thread_resources.safe_print(f'✅ {item}: No overlaps')
            return self._create_success_result(item)
    
    @staticmethod
    def _create_error_result(item: str, error: str) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            'item': item,
            'df_result': pd.DataFrame([]),
            'gdf_intr': None,
            'col_lbl': None,
            'overlaps': 0,
            'success': False,
            'error': error
        }
    
    @staticmethod
    def _create_success_result(item: str) -> Dict[str, Any]:
        """Create success result dictionary with no overlaps."""
        return {
            'item': item,
            'df_result': pd.DataFrame([]),
            'gdf_intr': None,
            'col_lbl': None,
            'overlaps': 0,
            'success': True,
            'error': None
        }


# ============================================================================
# PARALLEL ANALYSIS COORDINATOR
# ============================================================================

class ParallelAnalysisCoordinator:
    """Coordinates parallel processing of datasets."""
    
    def __init__(
        self,
        df_stat: pd.DataFrame,
        wkb_aoi: bytes,
        srid: int,
        gdf_aoi: gpd.GeoDataFrame,
        region: str,
        workspace: str,
        db_config: Dict[str, Any],
        sql_queries: Dict[str, str],
        max_workers: int = 4
    ):
        self.df_stat = df_stat
        self.wkb_aoi = wkb_aoi
        self.srid = srid
        self.gdf_aoi = gdf_aoi
        self.region = region
        self.workspace = workspace
        self.db_config = db_config
        self.sql = sql_queries
        self.max_workers = max_workers
        
        self.results = {}
        self.failed_datasets = []
        self.datasets_with_overlaps = []
    
    def run_analysis(self) -> None:
        """Run parallel analysis on all datasets."""
        processor = DatasetProcessor(self.db_config, self.sql)
        item_count = self.df_stat.shape[0]
        
        print(f'\nConfiguration:')
        print(f'  - Parallel workers: {self.max_workers}')
        print(f'  - Total datasets: {item_count}')
        print(f'  - Region: {self.region}')
        print('\n' + '='*80)
        print('PROCESSING DATASETS')
        print('='*80 + '\n')
        
        # Prepare dataset arguments
        dataset_args = [
            (index, row) 
            for index, row in self.df_stat.iterrows()
        ]
        
        # Process datasets in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dataset = {
                executor.submit(
                    processor.process_dataset,
                    index, row, self.df_stat, self.wkb_aoi,
                    self.srid, self.region, self.gdf_aoi
                ): row['Featureclass_Name(valid characters only)']
                for index, row in dataset_args
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
                    with _thread_resources.results_lock:
                        self.results[result['item']] = result['df_result']
                        
                        if not result['success']:
                            self.failed_datasets.append({
                                'item': result['item'],
                                'reason': result['error']
                            })
                        elif result['gdf_intr'] is not None and result['overlaps'] > 0:
                            # Generate map
                            gdf_simplified = GeometryProcessor.simplify_geometries(
                                result['gdf_intr'], 
                                tolerance=10
                            )
                            MapGenerator.create_status_map(
                                self.gdf_aoi,
                                gdf_simplified,
                                result['col_lbl'],
                                result['item'],
                                self.workspace
                            )
                            self.datasets_with_overlaps.append({
                                'item': result['item'],
                                'overlaps': result['overlaps']
                            })
                    
                    # Progress update
                    self._print_progress(completed, item_count, start_time)
                    
                except Exception as e:
                    _thread_resources.safe_print(f'❌ Critical error processing {dataset_name}: {e}')
                    with _thread_resources.results_lock:
                        self.failed_datasets.append({'item': dataset_name, 'reason': str(e)})
    
    def _print_progress(self, completed: int, total: int, start_time: float) -> None:
        """Print progress bar and timing information."""
        elapsed = timeit.default_timer() - start_time
        avg_time = elapsed / completed
        remaining = (total - completed) * avg_time
        
        progress_pct = int(50 * completed / total)
        progress_bar = '█' * progress_pct + '░' * (50 - progress_pct)
        
        with _thread_resources.print_lock:
            print(f'\n[{progress_bar}] {completed}/{total} ({completed*100//total}%)')
            print(f'   Time elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | '
                  f'Est. remaining: {int(remaining//60)}m {int(remaining%60)}s')
    
    def print_summary(self, total_time: float) -> None:
        """Print comprehensive analysis summary."""
        item_count = self.df_stat.shape[0]
        mins = int(total_time / 60)
        secs = int(total_time % 60)
        
        print('\n' + '='*80)
        print('PROCESSING SUMMARY')
        print('='*80)
        print(f'\n📊 Statistics:')
        print(f'   Total datasets processed: {item_count}')
        print(f'   Datasets with overlaps: {len(self.datasets_with_overlaps)}')
        print(f'   Datasets with no overlaps: {item_count - len(self.datasets_with_overlaps) - len(self.failed_datasets)}')
        print(f'   Failed datasets: {len(self.failed_datasets)}')
        
        print(f'\n⏱️  Performance:')
        print(f'   Total time: {mins}m {secs}s')
        print(f'   Average per dataset: {total_time/item_count:.1f}s')
        print(f'   Parallel workers used: {self.max_workers}')
        
        if self.datasets_with_overlaps:
            print(f'\n🔍 Datasets with overlaps:')
            sorted_overlaps = sorted(
                self.datasets_with_overlaps, 
                key=lambda x: x['overlaps'], 
                reverse=True
            )
            for ds in sorted_overlaps[:10]:
                print(f'   • {ds["item"]}: {ds["overlaps"]} overlap(s)')
            if len(self.datasets_with_overlaps) > 10:
                print(f'   ... and {len(self.datasets_with_overlaps) - 10} more')
        
        if self.failed_datasets:
            print(f'\n❌ Failed datasets:')
            for failed in self.failed_datasets[:10]:
                reason = failed["reason"]
                print(f'   • {failed["item"]}')
                print(f'     Reason: {reason[:80]}{"..." if len(reason) > 80 else ""}')
            if len(self.failed_datasets) > 10:
                print(f'   ... and {len(self.failed_datasets) - 10} more')
        else:
            print(f'\n✅ All datasets processed successfully!')
        
        print('\n' + '='*80)
        print(f'Output location: {self.workspace}')
        print('='*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the parallelized AST lite process."""
    start_time = timeit.default_timer()
    
    # Configuration
    workspace = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
    wksp_xls = os.path.join(workspace, 'input_spreadsheets')
    aoi_path = os.path.join(workspace, 'test_data', 'aoi_test_4.shp')
    out_wksp = os.path.join(workspace, 'outputs')
    
    # User inputs
    input_src = 'TANTALIS'  # Options: 'TANTALIS' or 'AOI'
    region = 'cariboo'
    max_workers = 4  # Adjust based on your system (recommended: 3-6)
    
    # TANTALIS parameters (if input_src == 'TANTALIS')
    file_nbr = '5408057'
    disp_id = 943829
    prcl_id = 977043
    # Database configuration
    db_config = {
        'oracle_host': 'bcgw.bcgov/idwprod1.bcgov',
        'oracle_user': os.getenv('bcgw_user'),
        'oracle_pwd': os.getenv('bcgw_pwd'),
        'pg_host': 'localhost',
        'pg_database': 'ast_local_datasets',
        'pg_user': 'postgres',
        'pg_pwd': os.getenv('PG_LCL_SUSR_PASS'),
        'pg_port': 5432
    }
    
    # Test connections
    print('Testing BCGW connection.')
    test_conn, test_cursor = connect_to_oracle(
        db_config['oracle_user'],
        db_config['oracle_pwd'],
        db_config['oracle_host']
    )
    test_cursor.close()
    test_conn.close()
    print("....Successfully connected to Oracle database")
    
    print('\nTesting PostGIS connection.')
    test_pg_conn, test_pg_cursor = connect_to_postgis(
        db_config['pg_host'],
        db_config['pg_database'],
        db_config['pg_user'],
        db_config['pg_pwd'],
        db_config['pg_port']
    )
    test_pg_cursor.close()
    test_pg_conn.close()
    print("....Successfully connected to PostGIS database")
    
    # Load SQL queries
    print('\nLoading SQL queries')
    sql = load_sql_queries()
    
    # Get AOI
    print('\nReading User inputs: AOI.')
    if input_src == 'AOI':
        print('....Reading the AOI file')
        gdf_aoi = GeometryProcessor.esri_to_gdf(aoi_path)
    elif input_src == 'TANTALIS':
        print(f'....input File Number: {file_nbr}')
        print(f'....input Disposition ID: {disp_id}')
        print(f'....input Parcel ID: {prcl_id}')
        
        bind_vars = {
            'file_nbr': file_nbr,
            'disp_id': disp_id,
            'parcel_id': prcl_id
        }
        
        print('....Querying TANTALIS for AOI geometry')
        temp_conn, temp_cursor = connect_to_oracle(
            db_config['oracle_user'],
            db_config['oracle_pwd'],
            db_config['oracle_host']
        )
        
        try:
            df_aoi = read_query(temp_conn, temp_cursor, sql['aoi'], bind_vars)
            
            if df_aoi.shape[0] < 1:
                raise Exception('Parcel not in TANTALIS. Please check inputs!')
            
            print('....Converting TANTALIS result to GeoDataFrame')
            if 'SHAPE' in df_aoi.columns:
                df_aoi['SHAPE'] = df_aoi['SHAPE'].apply(
                    lambda x: str(x) if x is not None else None
                )
            
            gdf_aoi = GeometryProcessor.df_to_gdf(df_aoi, 3005)
        finally:
            temp_cursor.close()
            temp_conn.close()
    else:
        raise ValueError('input_src must be "TANTALIS" or "AOI"')
    
    # Process multipart AOI
    if gdf_aoi.shape[0] > 1:
        print('....Converting multipart AOI to singlepart')
        gdf_aoi = GeometryProcessor.multipart_to_singlepart(gdf_aoi)
    
    # Extract WKB and SRID
    print('....Extracting WKB and SRID from AOI')
    wkb_aoi, srid = GeometryProcessor.get_wkb_srid(gdf_aoi)
    
    # Read configuration
    print('\nReading the AST datasets spreadsheet.')
    print(f'....Region is {region}')
    df_stat = DatasetConfig.read_spreadsheets(wksp_xls, region)
    
    print('\n' + '='*80)
    print('RUNNING PARALLEL ANALYSIS')
    print('='*80)
    
    # Run parallel analysis
    coordinator = ParallelAnalysisCoordinator(
        df_stat=df_stat,
        wkb_aoi=wkb_aoi,
        srid=srid,
        gdf_aoi=gdf_aoi,
        region=region,
        workspace=out_wksp,
        db_config=db_config,
        sql_queries=sql,
        max_workers=max_workers
    )
    
    coordinator.run_analysis()
    
    # Write results
    print('\n' + '='*80)
    print('WRITING RESULTS TO SPREADSHEET')
    print('='*80)
    ExcelReportWriter.write_report(coordinator.results, df_stat, out_wksp)
    print('✅ Results spreadsheet created')
    
    # Print summary
    finish_time = timeit.default_timer()
    total_time = finish_time - start_time
    coordinator.print_summary(total_time)


if __name__ == "__main__":
    main()