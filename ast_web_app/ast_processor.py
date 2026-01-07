"""
AST Processor Module

Wraps the original AST script logic for use with the web GUI.
Provides progress callbacks and result aggregation.

Author: Moez Labiadh - GeoBC

Created: 2026-01-06
"""

import os
import warnings
warnings.simplefilter(action='ignore')

from pathlib import Path
from typing import Dict, Callable, Optional
import pandas as pd

# Import all the classes from the core processing script
from ast_core import (
    OracleConnection,
    PostGISConnection,
    GeometryProcessor,
    DatasetConfig,
    OverlayAnalyzer,
    ExcelReportWriter,
    load_sql_queries
)


class ASTProcessor:
    """
    Main processor that wraps AST functionality for web GUI.
    Provides progress tracking and result aggregation.
    """
    
    def __init__(self, config: Dict, progress_callback: Optional[Callable] = None, 
                 cancellation_check: Optional[Callable] = None):
        """
        Initialize AST processor.
        
        Args:
            config: Configuration dictionary containing:
                - input_source: 'TANTALIS' or 'AOI'
                - region: region name
                - workspace: output directory
                - bcgw: Oracle connection details
                - postgis: PostGIS connection details
                - tantalis: TANTALIS parameters (if applicable)
                - aoi_file: AOI file path (if applicable)
            progress_callback: Function to call with progress updates
                               signature: callback(progress_percent, message)
            cancellation_check: Function that returns True if analysis should be cancelled
        """
        self.config = config
        self.progress_callback = progress_callback
        self.cancellation_check = cancellation_check
        self.oracle_conn = None
        self.postgis_conn = None
        self.results = {}
        self.failed_datasets = []
    
    def _update_progress(self, progress: int, message: str):
        """Update progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def _is_cancelled(self) -> bool:
        """Check if analysis has been cancelled."""
        if self.cancellation_check:
            return self.cancellation_check()
        return False
    
    def run(self) -> Dict:
        """
        Execute the full AST analysis.
        
        Returns:
            Dictionary containing:
                - total_datasets: number of datasets analyzed
                - conflicts_found: number of datasets with conflicts
                - conflict_counts: dict of conflict counts by dataset
                - conflicts_by_category: list of dicts for display
                - failed_datasets: number of failed datasets
                - output_file: path to Excel report
        """
        try:
            # Setup workspace
            workspace = Path(self.config['workspace'])
            workspace.mkdir(parents=True, exist_ok=True)
            
            # Create maps subdirectory
            maps_dir = workspace / 'maps'
            maps_dir.mkdir(exist_ok=True)
            
            # Connect to databases
            self._update_progress(5, "Connecting to Oracle/BCGW...")
            self.oracle_conn = self._connect_oracle()
            
            self._update_progress(10, "Connecting to PostGIS...")
            self.postgis_conn = self._connect_postgis()
            
            # Load SQL queries
            self._update_progress(15, "Loading SQL queries...")
            sql = load_sql_queries()
            
            # Get AOI
            self._update_progress(20, "Loading Area of Interest...")
            gdf_aoi, wkb_aoi, srid = self._load_aoi()
            
            # Read configuration
            self._update_progress(25, "Reading dataset configuration...")
            df_stat = self._load_dataset_config()
            
            # Run analysis
            self._update_progress(30, "Starting overlay analysis...")
            self._run_overlay_analysis(df_stat, wkb_aoi, srid, gdf_aoi, workspace, sql)
            
            # Write results
            self._update_progress(90, "Writing results to Excel...")
            output_file = self._write_results(df_stat, workspace)
            
            # Aggregate results
            self._update_progress(95, "Finalizing results...")
            results = self._aggregate_results(df_stat, output_file)
            
            self._update_progress(100, "Analysis complete!")
            
            return results
            
        except Exception as e:
            self._update_progress(0, f"Error: {str(e)}")
            raise
        finally:
            self._cleanup()
    
    def _connect_oracle(self) -> OracleConnection:
        """Establish Oracle connection."""
        bcgw = self.config['bcgw']
        return OracleConnection(
            bcgw['username'],
            bcgw['password'],
            bcgw['hostname']
        )
    
    def _connect_postgis(self) -> PostGISConnection:
        """Establish PostGIS connection."""
        pg = self.config['postgis']
        return PostGISConnection(
            pg['host'],
            pg['database'],
            pg['user'],
            pg['password']
        )
    
    def _load_aoi(self):
        """Load and process AOI geometry."""
        if self.config['input_source'] in ['UPLOAD', 'AOI', 'SHAPEFILE']:
            # Load from file (shapefile or feature class)
            aoi_path = self.config['aoi_file']
            gdf_aoi = GeometryProcessor.esri_to_gdf(aoi_path)
        else:
            # Query from TANTALIS
            tantalis = self.config['tantalis']
            
            bind_vars = {
                'file_nbr': tantalis['file_number'],
                'disp_id': tantalis['disposition_id'],
                'parcel_id': tantalis['parcel_id']
            }
            
            aoi_query = """
                SELECT SDO_UTIL.TO_WKTGEOMETRY(a.SHAPE) SHAPE
                FROM WHSE_TANTALIS.TA_CROWN_TENURES_SVW a
                WHERE a.CROWN_LANDS_FILE = :file_nbr
                    AND a.DISPOSITION_TRANSACTION_SID = :disp_id
                    AND a.INTRID_SID = :parcel_id
            """
            
            conn = self.oracle_conn.connection
            cursor = self.oracle_conn.cursor
            cursor.execute(aoi_query, bind_vars)
            names = [x[0] for x in cursor.description]
            rows = cursor.fetchall()
            df_aoi = pd.DataFrame(rows, columns=names)
            
            if df_aoi.shape[0] < 1:
                raise Exception('Parcel not found in TANTALIS. Please check inputs!')
            
            gdf_aoi = GeometryProcessor.df_to_gdf(df_aoi, 3005)
        
        # Convert multipart to singlepart if needed
        if gdf_aoi.shape[0] > 1:
            gdf_aoi = GeometryProcessor.multipart_to_singlepart(gdf_aoi)
        
        # Extract WKB and SRID
        wkb_aoi, srid = GeometryProcessor.get_wkb_srid(gdf_aoi)
        
        return gdf_aoi, wkb_aoi, srid
    
    def _load_dataset_config(self) -> pd.DataFrame:
        """Load dataset configuration spreadsheets."""
        # Default to the known path
        workspace_xls = self.config.get(
            'workspace_xls',
            r'W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\input_spreadsheets'
        )
        
        return DatasetConfig.read_spreadsheets(
            str(workspace_xls),
            self.config['region']
        )
    
    def _run_overlay_analysis(self, df_stat, wkb_aoi, srid, gdf_aoi, workspace, sql):
        """Run overlay analysis on all datasets."""
        analyzer = OverlayAnalyzer(
            self.oracle_conn,
            self.postgis_conn,
            sql
        )
        
        item_count = df_stat.shape[0]
        
        for index in df_stat.index:
            # Check for cancellation
            if self._is_cancelled():
                self._update_progress(0, "Analysis cancelled by user")
                raise Exception("Analysis cancelled by user")
            
            # Calculate progress (30% to 85% of total progress)
            progress = 30 + int((index / item_count) * 55)
            
            item = df_stat.loc[index, 'Featureclass_Name(valid characters only)']
            self._update_progress(
                progress,
                f"Analyzing dataset {index + 1} of {item_count}: {item}"
            )
            
            try:
                analyzer.analyze_dataset(
                    index,
                    df_stat,
                    wkb_aoi,
                    srid,
                    gdf_aoi,
                    str(workspace),
                    self.config['region']
                )
            except Exception as e:
                print(f"Failed to process {item}: {e}")
                analyzer.failed_datasets.append({
                    'item': item,
                    'reason': str(e)
                })
        
        # Store results
        self.results = analyzer.results
        self.failed_datasets = analyzer.failed_datasets
    
    def _write_results(self, df_stat, workspace) -> str:
        """Write results to Excel and return file path."""
        ExcelReportWriter.write_report(
            self.results,
            df_stat,
            str(workspace)
        )
        
        return str(workspace / 'AST_lite_TAB3.xlsx')
    
    def _aggregate_results(self, df_stat, output_file) -> Dict:
        """Aggregate results for web display."""
        # Count conflicts by dataset
        conflict_counts = {}
        conflicts_found = 0
        
        for item_name, result_df in self.results.items():
            count = result_df.shape[0]
            if count > 0:
                conflict_counts[item_name] = count
                conflicts_found += 1
        
        # Group by category
        df_res = df_stat[['Category', 'Featureclass_Name(valid characters only)']].copy()
        df_res.rename(
            columns={'Featureclass_Name(valid characters only)': 'item'},
            inplace=True
        )
        
        category_conflicts = {}
        for _, row in df_res.iterrows():
            category = row['Category']
            item = row['item']
            
            if item in conflict_counts:
                category_conflicts[category] = category_conflicts.get(category, 0) + conflict_counts[item]
        
        conflicts_by_category = [
            {'category': cat, 'count': count}
            for cat, count in category_conflicts.items()
        ]
        
        return {
            'total_datasets': df_stat.shape[0],
            'conflicts_found': conflicts_found,
            'conflict_counts': conflict_counts,
            'conflicts_by_category': conflicts_by_category,
            'failed_datasets': len(self.failed_datasets),
            'failed_details': self.failed_datasets,  # Include full details
            'output_file': output_file
        }
    
    def _cleanup(self):
        """Clean up database connections."""
        if self.oracle_conn:
            self.oracle_conn.close()
            print("Oracle connection closed")
        
        if self.postgis_conn:
            self.postgis_conn.close()
            print("PostGIS connection closed")