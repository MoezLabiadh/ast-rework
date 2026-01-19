"""
ast_core.py

Automatic Status Tool - LITE Version

Purpose:     This script checks for overlaps between an AOI and datasets
             specified in the AST datasets spreadsheets (common and region specific). 
             
Notes        The script supports AOIs in TANTALIS Crown Tenure spatial view 
             and User defined AOIs (shp, featureclass, kml/kmz).
               
             The script generates a spreadhseet of conflicts (TAB3) of the 
             standard AST reportand Interactive HTML maps showing the AOI and ovelappng features

            This version of the script uses postgis to process local datasets.
                             
Arguments:   - Output location (workspace)
             - DB credentials for Oracle/BCGW and PostGIS
             - Input source: TANTALIS OR AOI
             - Region (west coast, skeena...)
             - AOI: - ESRI shp or featureclass or KML/KMZ(AOI) OR
                    - TANTALIS File number
                    - TANTALIS Disposition ID
                    - TANTALIS Parcel ID

Author: Moez Labiadh - GeoBC

Created: 2025-12-23
Updated: 2025-01-19
"""

import warnings
warnings.simplefilter(action='ignore')

import os
import re
import timeit
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import oracledb
import psycopg2
import pandas as pd
import geopandas as gpd
from shapely import from_wkt, wkb

# Import the enhanced mapping module
import sys
sys.path.append(r"W:\srm\gss\sandbox\mlabiadh\git\ast-rework\ast_web_app")
from ast_mapping import MapGenerator


# ============================================================================
# DATABASE CONNECTION CLASSES
# ============================================================================

class DatabaseConnection:
    """Base class for database connections with context manager support."""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        return self.connection, self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self) -> None:
        """Close cursor and connection safely."""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                print(f"Warning: Error closing cursor: {e}")
        
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                print(f"Warning: Error closing connection: {e}")


class OracleConnection(DatabaseConnection):
    """Oracle database connection manager."""
    
    def __init__(self, username: str, password: str, hostname: str):
        super().__init__()
        self.username = username
        self.password = password
        self.hostname = hostname
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Oracle database."""
        try:
            self.connection = oracledb.connect(
                user=self.username,
                password=self.password,
                dsn=self.hostname
            )
            self.cursor = self.connection.cursor()
            print("....Successfully connected to Oracle database")
        except Exception as e:
            raise Exception(f'....Oracle connection failed! Error: {e}')


class PostGISConnection(DatabaseConnection):
    """PostGIS database connection manager."""
    
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        super().__init__()
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to PostGIS database."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            print("....Successfully connected to PostGIS database")
        except Exception as e:
            raise Exception(f'....PostGIS connection failed! Error: {e}')


# ============================================================================
# QUERY UTILITIES
# ============================================================================

def read_query(connection: Any, cursor: Any, query: str, bind_vars: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.
    
    Args:
        connection: Database connection object
        cursor: Database cursor object
        query: SQL query string
        bind_vars: Dictionary of bind variables
    
    Returns:
        DataFrame containing query results
    """
    cursor.execute(query, bind_vars)
    names = [x[0] for x in cursor.description]
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=names)


def load_sql_queries() -> Dict[str, str]:
    """
    Load all SQL query templates.
    
    Returns:
        Dictionary of SQL query strings
    """
    return {
        # Oracle queries
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
                CASE 
                    WHEN SDO_RELATE({geom_col}, 
                                    SDO_GEOMETRY(:wkb_aoi, :srid),
                                    'mask=ANYINTERACT') = 'TRUE'
                    THEN 'INTERSECT'
                    ELSE 'Within {radius} m'
                END AS RESULT,
                SDO_UTIL.TO_WKTGEOMETRY({geom_col}) SHAPE
            FROM {tab}
            WHERE SDO_WITHIN_DISTANCE({geom_col}, 
                                    SDO_GEOMETRY(:wkb_aoi, :srid),
                                    'distance={radius}') = 'TRUE'
                {def_query}
        """,
        
        # PostGIS queries
        # PostGIS overlay query optimization (similar to Oracle approach):
        # Uses ST_Intersects for direct overlaps (radius=0) - faster, no distance calculation
        # Uses ST_DWithin for buffer/proximity checks (radius>0) - required for distance filtering
        
        # Query for radius = 0 (direct overlap only)
        # ST_Intersects is faster than ST_DWithin with distance=0 because:
        # 1. No distance calculation overhead
        # 2. Can use spatial index more efficiently
        # 3. Short-circuits on first intersection found
        'postgis_overlay_zero_radius': """
            SELECT {cols},
                   'INTERSECT' AS result,
                   ST_AsText(geometry) AS shape
            FROM {schema}.{table}
            WHERE geometry && ST_GeomFromWKB(%s, %s)
              AND ST_Intersects(geometry, ST_GeomFromWKB(%s, %s))
            {def_query}
        """,
        
        # Query for radius > 0 (buffer/distance check)
        # ST_DWithin uses the spatial index efficiently for distance queries
        # The && operator provides a bounding box pre-filter for additional speed
        'postgis_overlay_with_radius': """
            SELECT {cols},
                   CASE 
                       WHEN ST_Intersects(geometry, ST_GeomFromWKB(%s, %s)) 
                       THEN 'INTERSECT'
                       ELSE 'Within {radius} m'
                   END AS result,
                   ST_AsText(geometry) AS shape
            FROM {schema}.{table}
            WHERE geometry && ST_Expand(ST_GeomFromWKB(%s, %s), {radius})
              AND ST_DWithin(geometry, ST_GeomFromWKB(%s, %s), {radius})
            {def_query}
        """
    }


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

class GeometryProcessor:
    """Handles geometry operations and conversions."""
    
    @staticmethod
    def read_spatial_file(aoi_path: str) -> gpd.GeoDataFrame:
        """
        Convert ESRI format vector or KML/KMZ to GeoDataFrame.
        """
        import fiona
        import zipfile
        import tempfile
        import shutil
        
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        
        # Convert to Path object for easier handling
        path = Path(aoi_path)
        aoi_path_lower = str(aoi_path).lower()
        
        # Handle Shapefile
        if '.shp' in aoi_path_lower:
            gdf = gpd.read_file(aoi_path)
        
        # Handle Feature Class (GDB)
        elif '.gdb' in aoi_path_lower:
            parts = aoi_path.split('.gdb')
            gdb = parts[0] + '.gdb'
            fc = os.path.basename(aoi_path)
            gdf = gpd.read_file(filename=gdb, layer=fc)
        
        # Handle KML
        elif aoi_path_lower.endswith('.kml'):
            try:
                layers = fiona.listlayers(aoi_path)
                if not layers:
                    raise ValueError(f'No layers found in KML file: {aoi_path}')
                
                layer_name = layers[0]
                print(f'....Reading KML layer: {layer_name}')
                gdf = gpd.read_file(aoi_path, driver='KML', layer=layer_name)
                
                if gdf.crs and gdf.crs.to_epsg() != 3005:
                    print(f'....Reprojecting from {gdf.crs.to_string()} to EPSG:3005')
                    gdf = gdf.to_crs(epsg=3005)
            except Exception as e:
                raise ValueError(f'Error reading KML file: {e}')
        
        # Handle KMZ (Robust method: Extract to temp then read)
        elif aoi_path_lower.endswith('.kmz'):
            tmp_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(aoi_path, 'r') as zip_ref:
                    # Find the first .kml file inside the KMZ
                    kml_files = [f for f in zip_ref.namelist() if f.lower().endswith('.kml')]
                    if not kml_files:
                        raise ValueError(f"No KML file found inside KMZ: {aoi_path}")
                    
                    # Extract the KML to the temp directory
                    extracted_kml_path = zip_ref.extract(kml_files[0], tmp_dir)
                
                # Use the same logic as the KML block
                layers = fiona.listlayers(extracted_kml_path)
                layer_name = layers[0]
                print(f'....Reading KMZ (extracted) layer: {layer_name}')
                
                gdf = gpd.read_file(extracted_kml_path, driver='KML', layer=layer_name)
                
                if gdf.crs and gdf.crs.to_epsg() != 3005:
                    print(f'....Reprojecting from {gdf.crs.to_string()} to EPSG:3005')
                    gdf = gdf.to_crs(epsg=3005)
                
            except Exception as e:
                raise ValueError(f'Error reading KMZ file: {e}')
            finally:
                # Always clean up the temporary directory
                shutil.rmtree(tmp_dir)
        
        else:
            raise ValueError(
                'Format not recognized. Please provide a shapefile (.shp), '
                'feature class (.gdb), KML (.kml), or KMZ (.kmz) file!'
            )
        
        # Validate that we have geometries
        if gdf.empty:
            raise ValueError(f'No features found in file: {aoi_path}')
        
        if 'geometry' not in gdf.columns and gdf.geometry is None:
            raise ValueError(f'No geometry column found in file: {aoi_path}')
        
        return gdf
    
    @staticmethod
    def df_to_gdf(df: pd.DataFrame, crs: int) -> gpd.GeoDataFrame:
        """
        Convert DataFrame with geometry column to GeoDataFrame.
        
        Args:
            df: DataFrame with SHAPE or shape column
            crs: EPSG code for coordinate reference system
        
        Returns:
            GeoDataFrame
        """
        # Determine geometry column name
        shape_col = 'SHAPE' if 'SHAPE' in df.columns else 'shape'
        if shape_col not in df.columns:
            raise ValueError("No geometry column found. Expected 'SHAPE' or 'shape'")
        
        shape_series = df[shape_col].astype(str)
        
        # Process geometries
        processed_wkts = [
            GeometryProcessor._process_geometry(wkt) 
            for wkt in shape_series
        ]
        
        # Create clean DataFrame without geometry column
        df_clean = df.drop(columns=[shape_col]).copy()
        
        # Create GeoDataFrame
        df_clean['geometry'] = gpd.GeoSeries.from_wkt(
            processed_wkts, 
            crs=f"EPSG:{crs}"
        )
        
        return gpd.GeoDataFrame(df_clean, geometry='geometry', crs=f"EPSG:{crs}")
    
    @staticmethod
    def _process_geometry(wkt_str: str) -> Optional[str]:
        """Process geometry string, linearizing curves if needed."""
        if wkt_str is None or not isinstance(wkt_str, str):
            return None
        
        # Check if geometry contains curves
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
        """
        Extract WKB and SRID from GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame
        
        Returns:
            Tuple of (WKB bytes, SRID integer)
        """
        srid = gdf.crs.to_epsg()
        geom = gdf['geometry'].iloc[0]
        
        # Handle 3D geometries
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
        """
        Read and combine common and region-specific dataset spreadsheets.
        
        Args:
            workspace_xls: Path to spreadsheet directory
            region: Region name (e.g., 'northeast')
        
        Returns:
            Combined DataFrame
        """
        common_xls = os.path.join(workspace_xls, 'one_status_common_datasets.xlsx')
        region_xls = os.path.join(
            workspace_xls, 
            f'one_status_{region.lower()}_specific.xlsx'
        )
        
        df_common = pd.read_excel(common_xls)
        df_region = pd.read_excel(region_xls)
        
        df_combined = pd.concat([df_common, df_region])
        df_combined.dropna(how='all', inplace=True)
        df_combined.reset_index(drop=True, inplace=True)
        
        return df_combined
    
    @staticmethod
    def get_table_columns(item_index: int, df_stat: pd.DataFrame) -> Tuple[str, str, str]:
        """
        Extract table name and column information from config.
        
        Args:
            item_index: Row index in config DataFrame
            df_stat: Configuration DataFrame
        
        Returns:
            Tuple of (table_name, columns_csv, label_column)
        """
        df_item = df_stat.loc[[item_index]].fillna('nan')
        
        table = df_item['Datasource'].iloc[0].strip()
        
        # Collect all field columns
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
        
        # Add map label field if not already included
        label_col = df_item['map_label_field'].iloc[0].strip()
        if label_col != 'nan' and label_col not in fields:
            fields.append(label_col)
        
        cols_csv = ','.join(fields) if fields else ''
        
        return table, cols_csv, label_col
    
    @staticmethod
    def get_definition_query(
        item_index: int, 
        df_stat: pd.DataFrame, 
        for_postgis: bool = False
    ) -> str:
        """
        Extract and format definition query from config.
        
        Args:
            item_index: Row index in config DataFrame
            df_stat: Configuration DataFrame
            for_postgis: Whether to format for PostGIS
        
        Returns:
            Formatted SQL WHERE clause
        """
        df_item = df_stat.loc[[item_index]].fillna('nan')
        def_query = df_item['Definition_Query'].iloc[0].strip()
        
        if def_query == 'nan' or not def_query:
            return ""
        
        def_query = def_query.replace('"', '')
        
        # Escape percent signs for PostGIS
        if for_postgis:
            def_query = def_query.replace('%', '%%')
        
        return f'AND ({def_query})'
    
    @staticmethod
    def get_buffer_distance(item_index: int, df_stat: pd.DataFrame) -> int:
        """
        Extract buffer distance from config.
        
        Args:
            item_index: Row index in config DataFrame
            df_stat: Configuration DataFrame
        
        Returns:
            Buffer distance in meters
        """
        df_item = df_stat.loc[[item_index]].fillna(0)
        df_item['Buffer_Distance'] = df_item['Buffer_Distance'].astype(int)
        return df_item['Buffer_Distance'].iloc[0]


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
    def get_geometry_column(
        connection: Any, 
        cursor: Any, 
        table: str, 
        geom_query: str
    ) -> str:
        """Get geometry column name for Oracle table."""
        parts = table.split('.')
        bind_vars = {'owner': parts[0].strip(), 'tab_name': parts[1].strip()}
        df = read_query(connection, cursor, geom_query, bind_vars)
        return df['GEOM_NAME'].iloc[0]
    
    @staticmethod
    def get_srid(
        connection: Any, 
        cursor: Any, 
        table: str, 
        geom_col: str, 
        srid_query: str
    ) -> Optional[int]:
        """Get SRID for Oracle table."""
        try:
            query = srid_query.format(tab=table, geom_col=geom_col)
            df = read_query(connection, cursor, query, {})
            
            if df.empty or df.shape[0] == 0:
                print(f'.......WARNING: Table {table} is empty, cannot determine SRID')
                return None
            
            return df['SP_REF'].iloc[0]
        except (IndexError, Exception) as e:
            print(f'.......WARNING: Cannot determine SRID for {table}: {e}')
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
        except Exception as e:
            print(f'.......ERROR retrieving columns for {table}: {e}')
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
        
        print('.......applying SDO_ARC_DENSIFY and RECTIFY_GEOMETRY fix')
        print('.......Note: Curved geometries will be densified, invalid geometries will be rectified')
        
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
        print(f'.......Applying coordinate transformation (table SRID: {srid_t})')
        
        # Transform in SDO_GEOM.SDO_DISTANCE
        query = query.replace(
            'SDO_GEOMETRY(:wkb_aoi, :srid), 0.5)',
            'SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t), 0.5)'
        )
        
        # Transform in SDO_WITHIN_DISTANCE
        query = query.replace(
            'SDO_GEOMETRY(:wkb_aoi, :srid),',
            'SDO_CS.TRANSFORM(SDO_GEOMETRY(:wkb_aoi, :srid), :srid, :srid_t),'
        )
        
        # Transform output geometry
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
        
        # Clean table name for PostgreSQL
        table_name = table_name.lower()
        table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
        table_name = re.sub(r'_+', '_', table_name)
        table_name = table_name.strip('_')
        
        # Ensure doesn't start with digit
        if table_name and table_name[0].isdigit():
            table_name = 't_' + table_name
        
        # Truncate if too long
        if len(table_name) > 50:
            original_length = len(table_name)
            table_name = table_name[:50].rstrip('_')
            print(f"    Note: Table name truncated from {original_length} to {len(table_name)} chars")
        
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
            print(f'.......ERROR checking if PostGIS table exists: {e}')
            # Rollback the transaction to recover from error state
            try:
                cursor.connection.rollback()
            except:
                pass
            return False
    
    @staticmethod
    def get_columns(connection: Any, cursor: Any, schema: str, table: str) -> List[str]:
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
            print(f'.......ERROR retrieving PostGIS columns for {schema}.{table}: {e}')
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
        """
        Validate that requested columns exist in dataset.
        
        Args:
            cols: Comma-separated column names
            available_cols: List of available columns
            item: Dataset item name
            table: Table name
            is_postgis: Whether this is a PostGIS table
        
        Returns:
            Tuple of (validated_columns_csv, missing_columns_list)
        """
        missing_cols = []
        
        # Parse requested columns
        requested = [c.strip() for c in cols.split(',')] if cols else []
        
        # Normalize case
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
        
        # Check for missing columns
        for col in requested:
            if col not in available_cols and col != objectid_col:
                missing_cols.append(col)
        
        # Handle fallback if no valid columns
        if len(missing_cols) == len(requested) or not requested:
            if objectid_col in available_cols:
                print(f'.......WARNING: No valid requested columns, using {objectid_col}')
                return objectid_col, missing_cols
            
            valid_available = [col for col in available_cols if col not in excluded_cols]
            if valid_available:
                print(f'.......WARNING: No valid requested columns, using {valid_available[0]}')
                return valid_available[0], missing_cols
            else:
                print('.......WARNING: No valid columns available')
                return objectid_col, missing_cols
        
        # Remove missing columns
        valid_cols = [c for c in requested if c not in missing_cols]
        
        if missing_cols:
            print(f'.......WARNING: Missing columns in {table}: {", ".join(missing_cols)}')
            print(f'.......Available columns: {", ".join(sorted(available_cols)[:10])}...')
        
        return ','.join(valid_cols) if valid_cols else objectid_col, missing_cols


# ============================================================================
# EXCEL REPORT GENERATION
# ============================================================================

class ExcelReportWriter:
    """Writes analysis results to Excel format with improved formatting."""
    
    @staticmethod
    def write_report(
        results: Dict[str, pd.DataFrame],
        df_stat: pd.DataFrame,
        workspace: str
    ) -> None:
        """
        Write results to Excel spreadsheet.

        Args:
            results: Dictionary of dataset results
            df_stat: Configuration DataFrame
            workspace: Output workspace directory
        """
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
                    
                    # Drop result column (case-insensitive)
                    result_col = None
                    if 'RESULT' in result_df.columns:
                        result_col = 'RESULT'
                    elif 'result' in result_df.columns:
                        result_col = 'result'
                    
                    if result_col:
                        result_df = result_df.drop(result_col, axis=1)
                    
                    # Combine columns into single result string
                    result_df['Result'] = result_df[result_df.columns].apply(
                        lambda r: '; '.join(r.values.astype(str)), 
                        axis=1
                    )
                    
                    # Add row for each conflict
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
        
        # Forward-fill missing Category values
        # Replace empty strings with NaN first, then forward fill
        df_res['Category'] = df_res['Category'].replace('', pd.NA)
        df_res['Category'] = df_res['Category'].ffill()
        

        # Fill empty "List of conflicts" with "No overlaps"
        df_res['List of conflicts'] = df_res['List of conflicts'].replace('', 'No overlaps')
        
        # Write to Excel
        filename = os.path.join(workspace, 'AST_lite_TAB3.xlsx')
        sheetname = 'Conflicts & Constraints'
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, sheet_name=sheetname, index=False, startrow=0, startcol=0)
            
            workbook = writer.book
            worksheet = writer.sheets[sheetname]
            
            #Format columns
            txt_format = workbook.add_format({'text_wrap': True})
            lnk_format = workbook.add_format({'underline': True, 'font_color': 'blue'})
            
            # Conditional formatting for "No overlaps" text
            no_overlaps_format = workbook.add_format({
                'font_color': '#FF8C00',  # Dark orange
                'text_wrap': True
            })
            
            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, 1, 60)
            worksheet.set_column(2, 2, 80, txt_format)
            worksheet.set_column(3, 3, 20)
            
            # Conditional formatting for hyperlinks
            worksheet.conditional_format(
                f'D2:D{df_res.shape[0] + 1}',
                {
                    'type': 'cell',
                    'criteria': 'equal to',
                    'value': '"View Map"',
                    'format': lnk_format
                }
            )
            
            # Apply dark orange format to all "No overlaps" cells in column C
            worksheet.conditional_format(
                f'C2:C{df_res.shape[0] + 1}',
                {
                    'type': 'cell',
                    'criteria': 'equal to',
                    'value': '"No overlaps"',
                    'format': no_overlaps_format
                }
            )
            
            # Add table
            col_names = [{'header': col_name} for col_name in df_res.columns]
            worksheet.add_table(
                0, 0, df_res.shape[0] + 1, df_res.shape[1] - 1,
                {'columns': col_names}
            )


# ============================================================================
# MAIN ANALYSIS ENGINE
# ============================================================================

class OverlayAnalyzer:
    """Main engine for running overlay analysis."""
    
    def __init__(
        self,
        oracle_conn: OracleConnection,
        postgis_conn: PostGISConnection,
        sql_queries: Dict[str, str],
        gdf_aoi: gpd.GeoDataFrame,
        df_stat: pd.DataFrame,
        workspace: str
    ):
        self.oracle_conn = oracle_conn
        self.postgis_conn = postgis_conn
        self.sql = sql_queries
        self.gdf_aoi = gdf_aoi
        self.df_stat = df_stat
        self.workspace = workspace
        self.results = {}
        self.failed_datasets = []
        
        # Initialize the enhanced MapGenerator
        print('\nInitializing Map Generator')
        self.map_generator = MapGenerator(
            gdf_aoi=gdf_aoi,
            workspace=workspace,
            df_stat=df_stat
        )
        self.map_generator.initialize_all_layers_map()
    
    def analyze_dataset(
        self,
        item_index: int,
        wkb_aoi: bytes,
        srid: int,
        region: str
    ) -> None:
        """
        Analyze single dataset for overlaps with AOI.
        
        Args:
            item_index: Row index in config DataFrame
            wkb_aoi: WKB representation of AOI
            srid: SRID of AOI
            region: Region name for PostGIS schema
        """
        item = self.df_stat.loc[item_index, 'Featureclass_Name(valid characters only)']
        
        # Get category, handling NaN values
        category = ''
        if 'Category' in self.df_stat.columns:
            cat_value = self.df_stat.loc[item_index, 'Category']
            if pd.notna(cat_value):
                category = str(cat_value)
        
        try:
            # Get configuration
            print('.....getting table and column names')
            table, cols, col_lbl = DatasetConfig.get_table_columns(item_index, self.df_stat)
            
            print('.....getting definition query (if any)')
            def_query = DatasetConfig.get_definition_query(item_index, self.df_stat, for_postgis=False)
            
            print('.....getting buffer distance (if any)')
            radius = DatasetConfig.get_buffer_distance(item_index, self.df_stat)
            
            # Determine if Oracle or PostGIS
            is_oracle = table.startswith('WHSE') or table.startswith('REG')
            
            print('.....running Overlay Analysis.')
            
            if is_oracle:
                df_result = self._analyze_oracle_dataset(
                    table, cols, col_lbl, def_query, radius, wkb_aoi, srid
                )
            else:
                df_result = self._analyze_postgis_dataset(
                    table, cols, col_lbl, def_query, radius, wkb_aoi, srid, region
                )
            
            # Build list of non-geometry columns actually returned
            cols_list = [
                c for c in df_result.columns
                if c.lower() not in ('shape', 'geometry')
            ]

            # Remove duplicates while preserving order
            cols_list = list(dict.fromkeys(cols_list))

            if not cols_list and not df_result.empty:
                print(f'.......WARNING: No valid columns found in results for {item}')
                self.results[item] = pd.DataFrame([])
                return

            df_all_res = df_result[cols_list] if not df_result.empty else df_result

            ov_nbr = df_all_res.shape[0]
            print(f'.....number of overlaps: {ov_nbr}')
            
            # Store results
            self.results[item] = df_all_res
            
            # Generate map if overlaps found
            if ov_nbr > 0:
                self._generate_map(
                    df_result, cols, col_lbl, item, 
                    category, table, is_oracle
                )
        
        except Exception as e:
            print(f'.......ERROR processing dataset {item}: {e}')
            self.failed_datasets.append({'item': item, 'reason': str(e)})
            self.results[item] = pd.DataFrame([])
            self.results[item] = pd.DataFrame([])
    
    def _analyze_oracle_dataset(
        self,
        table: str,
        cols: str,
        col_lbl: str,
        def_query: str,
        radius: int,
        wkb_aoi: bytes,
        srid: int
    ) -> pd.DataFrame:
        """Analyze Oracle/BCGW dataset."""
        conn, cursor = self.oracle_conn.connection, self.oracle_conn.cursor
        
        # Get geometry column
        geom_col = OracleUtils.get_geometry_column(conn, cursor, table, self.sql['geomCol'])
        
        # Get SRID
        srid_t = OracleUtils.get_srid(conn, cursor, table, geom_col, self.sql['srid'])
        if srid_t is None:
            print(f'.......SKIPPING dataset - table is empty')
            raise Exception('Table is empty')
        
        # Validate columns
        print('.....validating columns')
        available_cols = OracleUtils.get_columns(conn, cursor, table)
        if not available_cols:
            print(f'.......SKIPPING dataset - could not retrieve table columns')
            raise Exception('Could not retrieve table columns')
        
        cols_upper = ColumnValidator.convert_to_uppercase(cols)
        validated_cols, _ = ColumnValidator.validate_columns(
            cols_upper, available_cols, table, table, is_postgis=False
        )
        
        # Clean up def_query - ensure it starts with AND if not empty
        clean_def_query = def_query.strip()
        if clean_def_query and not clean_def_query.upper().startswith('AND'):
            clean_def_query = 'AND ' + clean_def_query
        
        # Choose appropriate query based on radius parameter
        # radius=0: Use SDO_RELATE for direct overlaps (faster for exact intersection)
        # radius>0: Use SDO_WITHIN_DISTANCE for buffer/proximity checks
        if radius == 0:
            query_template = self.sql['oracle_overlay_zero_radius']
            print(f'.....using SDO_RELATE for direct overlap (radius=0)')
        else:
            query_template = self.sql['oracle_overlay_with_radius']
            print(f'.....using SDO_WITHIN_DISTANCE for buffer check (radius={radius}m)')
        
        # Build query
        query = query_template.format(
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
        
        # Apply geometry fix for problematic tables (densify curves, rectify invalid geometries)
        query = OracleUtils.apply_geometry_fix(query, table, geom_col)
        
        # Execute query
        df_all = read_query(conn, cursor, query, bind_vars)
        
        return df_all
    
    def _analyze_postgis_dataset(
        self,
        table: str,
        cols: str,
        col_lbl: str,
        def_query: str,
        radius: int,
        wkb_aoi: bytes,
        srid: int,
        region: str
    ) -> pd.DataFrame:
        """Analyze PostGIS dataset."""
        conn, cursor = self.postgis_conn.connection, self.postgis_conn.cursor
        
        # Get table name
        table_name = PostGISUtils.get_table_name_from_datasource(table)
        schema = region.lower()
        
        print(f'.......Using PostGIS table: {schema}.{table_name}')
        
        # Check if table exists
        print('.....checking if table exists in PostGIS')
        if not PostGISUtils.table_exists(cursor, schema, table_name):
            print(f'.......SKIPPING dataset - table {schema}.{table_name} not found in PostGIS')
            raise Exception(f'Table not found in PostGIS: {schema}.{table_name}')
        
        # Validate columns
        print('.....validating columns')
        available_cols = PostGISUtils.get_columns(conn, cursor, schema, table_name)
        if not available_cols:
            print(f'.......SKIPPING dataset - table exists but no columns found')
            raise Exception(f'No columns found in PostGIS table {schema}.{table_name}')
        
        cols_lower = ColumnValidator.convert_to_lowercase(cols)
        validated_cols, _ = ColumnValidator.validate_columns(
            cols_lower, available_cols, table_name, f'{schema}.{table_name}', is_postgis=True
        )
        
        # Get PostGIS definition query - don't wrap it again, just clean it
        df_temp = pd.DataFrame([{'Definition_Query': def_query if def_query.strip() else 'nan'}])
        pg_def_query_raw = DatasetConfig.get_definition_query(0, df_temp, for_postgis=True)
        
        # Remove the "AND " prefix if it was added, we'll add it in the query template
        pg_def_query = pg_def_query_raw.replace('AND (', '(').replace('AND(', '(') if pg_def_query_raw else ''
        
        # Only add the AND prefix if there's actually a query
        if pg_def_query and pg_def_query.strip():
            pg_def_query = 'AND ' + pg_def_query
        
        # Choose appropriate query based on radius parameter
        # radius=0: Use ST_Intersects for direct overlaps (faster, no distance calculation)
        # radius>0: Use ST_DWithin for buffer/proximity checks
        if radius == 0:
            query_template = self.sql['postgis_overlay_zero_radius']
            print(f'.....using ST_Intersects for direct overlap (radius=0)')
        else:
            query_template = self.sql['postgis_overlay_with_radius']
            print(f'.....using ST_DWithin for buffer check (radius={radius}m)')
        
        # Build query
        query = query_template.format(
            cols=validated_cols,
            schema=schema,
            table=table_name,
            radius=radius,
            def_query=pg_def_query
        )
        
        # Execute query with appropriate parameters based on query type
        try:
            if radius == 0:
                # Zero radius query: needs 4 parameters (2 for && bbox filter, 2 for ST_Intersects)
                cursor.execute(query, (
                    psycopg2.Binary(wkb_aoi),
                    int(srid),
                    psycopg2.Binary(wkb_aoi),
                    int(srid)
                ))
            else:
                # Non-zero radius query: needs 6 parameters 
                # (2 for CASE ST_Intersects, 2 for && bbox filter, 2 for ST_DWithin)
                cursor.execute(query, (
                    psycopg2.Binary(wkb_aoi),
                    int(srid),
                    psycopg2.Binary(wkb_aoi),
                    int(srid),
                    psycopg2.Binary(wkb_aoi),
                    int(srid)
                ))
            
            # Fetch results
            names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            df_all = pd.DataFrame(rows, columns=names)
            
            return df_all
            
        except psycopg2.Error as e:
            print(f'.......ERROR: PostGIS query failed: {e}')
            # Rollback the transaction to recover from error state
            try:
                cursor.connection.rollback()
            except:
                pass
            raise Exception(f'PostGIS query error: {str(e)}')
    
    def _generate_map(
        self,
        df_result: pd.DataFrame,
        cols: str,
        col_lbl: str,
        item: str,
        category: str,
        data_source: str,
        is_oracle: bool
    ) -> None:
        """Generate map for dataset with overlaps using the enhanced MapGenerator."""
        print('.....generating a map.')
        
        # Check if DataFrame has geometry before converting
        if df_result.empty:
            print('.......WARNING: Cannot create map - no results')
            return
        
        # Check for geometry column
        has_geom = False
        if is_oracle and 'SHAPE' in df_result.columns:
            has_geom = True
        elif not is_oracle and 'shape' in df_result.columns:
            has_geom = True
        
        if not has_geom:
            print('.......WARNING: Cannot create map - no geometry column in results')
            return
        
        # Convert to GeoDataFrame
        try:
            gdf_intersect = GeometryProcessor.df_to_gdf(df_result, 3005)
        except Exception as e:
            print(f'.......WARNING: Cannot create map - geometry conversion failed: {e}')
            return
        
        # Determine label column
        cols_list = [c.strip() for c in cols.split(',') if c.strip()]
        if is_oracle:
            col_lbl_use = col_lbl.upper() if col_lbl != 'nan' else (cols_list[0].upper() if cols_list else 'OBJECTID')
        else:
            col_lbl_use = col_lbl.lower() if col_lbl != 'nan' else (cols_list[0].lower() if cols_list else 'objectid')
        
        if col_lbl_use not in gdf_intersect.columns:
            # Fallback to first available column (excluding geometry)
            available_cols = [c for c in gdf_intersect.columns if c not in ['geometry', 'SHAPE', 'shape']]
            if available_cols:
                col_lbl_use = available_cols[0]
            else:
                print('.......WARNING: No valid label column found for map')
                return
        
        # Convert label column to string
        if col_lbl_use in gdf_intersect.columns:
            gdf_intersect[col_lbl_use] = gdf_intersect[col_lbl_use].astype(str)
        
        # Convert datetime columns
        for col in gdf_intersect.columns:
            if gdf_intersect[col].dtype == 'datetime64[ns]':
                gdf_intersect[col] = gdf_intersect[col].astype(str)
        
        # Simplify geometries for web display
        gdf_intersect_s = GeometryProcessor.simplify_geometries(gdf_intersect, tolerance=10)
        
        # Create individual map and add to all-layers map
        self.map_generator.create_individual_map(
            gdf_intersect=gdf_intersect_s,
            label_col=col_lbl_use,
            item_name=item,
            category=category,
            data_source=data_source
        )
    
    def finalize_maps(self) -> None:
        """Save the all-layers combined map. Call this after all datasets are processed."""
        self.map_generator.save_all_layers_map()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the AST lite process."""
    start_time = timeit.default_timer()
    
    # Configuration
    workspace = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
    wksp_xls = os.path.join(workspace, 'input_spreadsheets')
    aoi = os.path.join(workspace, 'test_data', 'aoi_test_3.shp')
    out_wksp = os.path.join(workspace, 'outputs')
    
    # User inputs
    input_src = 'AOI'  # Options: 'TANTALIS' or 'AOI'
    region = 'west_coast'
    
    # TANTALIS parameters (if input_src == 'TANTALIS')
    file_nbr = '8016020'
    disp_id = 944865
    prcl_id = 978397
    
    oracle_conn = None
    postgis_conn = None
    
    try:
        # Connect to databases
        print('Connecting to BCGW.')
        hostname = 'bcgw.bcgov/idwprod1.bcgov'
        bcgw_user = os.getenv('bcgw_user')
        bcgw_pwd = os.getenv('bcgw_pwd')
        oracle_conn = OracleConnection(bcgw_user, bcgw_pwd, hostname)
        
        print('\nConnecting to PostGIS.')
        pg_host = 'localhost'
        pg_database = 'ast_local_datasets'
        pg_user = 'postgres'
        pg_pwd = os.getenv('PG_LCL_SUSR_PASS')
        postgis_conn = PostGISConnection(pg_host, pg_database, pg_user, pg_pwd)
        
        # Load SQL queries
        print('\nLoading SQL queries')
        sql = load_sql_queries()
        
        # Get AOI
        print('\nReading User inputs: AOI.')
        if input_src == 'AOI':
            print('....Reading the AOI file')
            gdf_aoi = GeometryProcessor.read_spatial_file(aoi)
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
            df_aoi = read_query(
                oracle_conn.connection,
                oracle_conn.cursor,
                sql['aoi'],
                bind_vars
            )
            
            if df_aoi.shape[0] < 1:
                raise Exception('Parcel not in TANTALIS. Please check inputs!')
            
            print('....Converting TANTALIS result to GeoDataFrame')
            gdf_aoi = GeometryProcessor.df_to_gdf(df_aoi, 3005)
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
        
        # Run analysis
        print('\nRunning the analysis.')
        analyzer = OverlayAnalyzer(
            oracle_conn=oracle_conn,
            postgis_conn=postgis_conn,
            sql_queries=sql,
            gdf_aoi=gdf_aoi,
            df_stat=df_stat,
            workspace=out_wksp
        )
        
        item_count = df_stat.shape[0]
        counter = 1
        
        for index in df_stat.index:
            item = df_stat.loc[index, 'Featureclass_Name(valid characters only)']
            print(f'\n****working on item {counter} of {item_count}: {item}***')
            
            analyzer.analyze_dataset(index, wkb_aoi, srid, region)
            counter += 1
        
        # Finalize maps (save the all-layers combined map)
        print('\nFinalizing maps')
        analyzer.finalize_maps()
        
        # Write results
        print('\nWriting Results to spreadsheet')
        ExcelReportWriter.write_report(analyzer.results, df_stat, out_wksp)
        
        # Print summary
        if analyzer.failed_datasets:
            print('\n' + '=' * 80)
            print('SUMMARY: The following datasets failed to process:')
            print('=' * 80)
            for failed in analyzer.failed_datasets:
                print(f"  - {failed['item']}")
                print(f"    Reason: {failed['reason']}")
            print(f'\nTotal failed: {len(analyzer.failed_datasets)} out of {item_count}')
            print('=' * 80)
        else:
            print('\n' + '=' * 80)
            print('SUCCESS: All datasets processed without errors!')
            print('=' * 80)
    
    finally:
        # Clean up connections
        if oracle_conn:
            oracle_conn.close()
            print('\nOracle connection closed.')
        if postgis_conn:
            postgis_conn.close()
            print('PostGIS connection closed.')
    
    # Print timing
    finish_time = timeit.default_timer()
    elapsed_sec = round(finish_time - start_time)
    mins = int(elapsed_sec / 60)
    secs = int(elapsed_sec % 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')


if __name__ == "__main__":
    main()