"""
Inventory of Local datasets used in the AST.
"""
import warnings
warnings.simplefilter(action='ignore')

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from datetime import datetime
import timeit


def esri_to_gdf(file_path):
    """Returns a Geopandas file (gdf) based on 
       an ESRI format vector (shp or featureclass/gdb)"""
    if '.shp' in file_path.lower():
        gdf = gpd.read_file(file_path)
    elif '.gdb' in file_path:
        l = file_path.split('.gdb')
        gdb = l[0] + '.gdb'
        fc = os.path.basename(file_path)
        gdf = gpd.read_file(filename=gdb, layer=fc)
    else:
        raise Exception('Format not recognized. Please provide a shp or featureclass (gdb)!')
    return gdf


def analyze_datasource(file_path):
    try:
        # Determine datasource type and read file
        gdf = esri_to_gdf(file_path)
        ext = Path(file_path).suffix.lower()
        datasource_type = "shp" if ".shp" in ext else "featureclass"

        # Extract metadata
        geometry_type = gdf.geom_type.unique().tolist()
        nbr_of_rows = len(gdf)
        nbr_of_columns = len(gdf.columns)

        if datasource_type == "shp":
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            latest_edit_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            gdb_path = file_path.split('.gdb')[0] + '.gdb'
            file_size = 9999
            latest_edit_date = datetime.fromtimestamp(os.path.getmtime(gdb_path)).strftime('%Y-%m-%d %H:%M:%S')

        return {
            "Datasource Status": "Valid",
            "Datasource Type": datasource_type,
            "Geometry Type": geometry_type,
            "Number of Rows": nbr_of_rows,
            "Number of Columns": nbr_of_columns,
            "File Size (MB)": round(file_size, 2),
            "Latest Edit Date": latest_edit_date
        }
    except Exception as e:
        return {"Datasource Status": "Invalid", "Error": str(e)}


def create_inventory(input_file):
    # Load the Excel file
    df = pd.read_excel(input_file)
    df.rename(columns={
        "Featureclass_Name(valid characters only)": "Dataset"},
        inplace=True)

    # Prepare an output list
    inventory = []
    count = len(df)
    inventory_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for i, row in df.iterrows():
        print(f"...reading dataset {i+1} of {count}: {row['Dataset']}")
        datasource = row['Datasource']
        datasource = datasource.strip() if isinstance(datasource, str) else datasource  # Handle paths with extra spaces

        # Skip datasources that are empty and BCGW datasets
        if pd.isna(datasource) or datasource.startswith("WHSE") or datasource.startswith("REG"):
            continue

        # Analyze the datasource
        metadata = analyze_datasource(datasource)

        # Append metadata to the output list
        inventory.append({
            "Inventory Date": inventory_date,
            "Dataset": row['Dataset'],
            **metadata
        })

    # Convert the inventory to a DataFrame
    inventory_df = pd.DataFrame(inventory)

    return inventory_df


def generate_report(workspace, df_list, sheet_list, filename):
    """ Exports dataframes to multi-tab excel spreadsheet"""
    file_name = os.path.join(workspace, filename + '.xlsx')
    import xlsxwriter
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe = dataframe.reset_index(drop=True)
        dataframe.index = dataframe.index + 1

        dataframe.to_excel(writer, sheet_name=sheet, index=False, startrow=0, startcol=0)

        worksheet = writer.sheets[sheet]

        worksheet.set_column(0, dataframe.shape[1], 20)

        col_names = [{'header': col_name} for col_name in dataframe.columns[1:-1]]
        col_names.insert(0, {'header': dataframe.columns[0], 'total_string': 'Total'})
        col_names.append({'header': dataframe.columns[-1], 'total_function': 'count'})

        worksheet.add_table(0, 0, dataframe.shape[0] + 1, dataframe.shape[1] - 1, {
            'total_row': True,
            'columns': col_names})

    writer.close()


if __name__ == "__main__":
    start_t = timeit.default_timer()
    # Input AST spreadsheets
    in_loc = r'P:\corp\script_whse\python\Utility_Misc\Ready\statusing_tools_arcpro\statusing_input_spreadsheets'
    in_files= {
        'RWC': os.path.join(in_loc, 'one_status_west_coast_specific.xlsx'),
        'RSC': os.path.join(in_loc, 'one_status_south_coast_specific.xlsx'),
        'RTO': os.path.join(in_loc, 'one_status_thompson_okanagan_specific.xlsx'),
        'RKB': os.path.join(in_loc, 'one_status_kootenay_boundary_specific.xlsx'),
        'RCB': os.path.join(in_loc, 'one_status_cariboo_specific.xlsx'),
        'RSK': os.path.join(in_loc, 'one_status_skeena_specific.xlsx'),
        'ROM': os.path.join(in_loc, 'one_status_omineca_specific.xlsx'),
        'RNO': os.path.join(in_loc, 'one_status_northeast_specific.xlsx')   
    }

    # Run inventory
    invs = {}
    total = len(in_files)
    count = 1

    for k, v in in_files.items():
        print(f'\nCreating inventory {count} of {total}: {k}')
        df_inv = create_inventory(v)
        if df_inv.shape[0] > 0:
            invs[k] = df_inv
        count += 1

    # Save the inventory spreadsheet
    out_loc = r'W:\lwbc\visr\Workarea\moez_labiadh\STATUSING\ast_rework\local_datasets\inventory'
    date = datetime.now().strftime("%Y%m%d")
    out_file = f'{date}_inventory_local_datasets'
    generate_report(
        out_loc,
        invs.values(),
        invs.keys(),
        out_file
    )

    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
