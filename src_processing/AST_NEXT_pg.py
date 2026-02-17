"""
AST_NEXT_pg.py
(standalone testing script)

AST-NEXT: Automatic Status Tool - Next Generation!

Purpose:     Standalone entry point for testing ast_core.py locally.
             Imports all logic from ast_web_app/ast_core.py and
             ast_web_app/ast_mapping.py, and provides a main() function
             with hardcoded test inputs.

Notes        The script supports AOIs in TANTALIS Crown Tenure spatial view
             and User defined AOIs (shp, featureclass, kml/kmz).

             The script generates a spreadhseet of conflicts (TAB3) of the
             standard AST report and Interactive HTML maps showing the AOI
             and overlapping features.

             This version of the script uses PostGIS to process local datasets.

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
Updated: 2026-02-17
"""

import warnings
warnings.simplefilter(action='ignore')

import os
import sys
import timeit
from pathlib import Path

# Add ast_web_app to path so we can import from it directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ast_web_app'))

from ast_core import (
    OracleConnection,
    PostGISConnection,
    read_query,
    load_sql_queries,
    GeometryProcessor,
    DatasetConfig,
    OverlayAnalyzer,
    ExcelReportWriter,
)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the AST process."""
    start_time = timeit.default_timer()

    # Configuration
    workspace = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
    wksp_xls = os.path.join(workspace, 'input_spreadsheets')
    aoi = os.path.join(workspace, 'test_data', 'aoi_test_3.shp')
    out_wksp = os.path.join(workspace, 'outputs')

    # User inputs
    input_src = 'AOI'  # Options: 'TANTALIS' or 'AOI'
    region = 'west_coast'

    create_maps = True      # Change to False to disable maps
    create_datasets = False  # Change to True to export overlap results as GeoPackage

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
            workspace=out_wksp,
            create_maps=create_maps,
            create_datasets=create_datasets
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
        ExcelReportWriter.write_report(
            analyzer.results,
            df_stat,
            out_wksp,
            create_maps=create_maps
        )

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
