"""
app.py

AST LITE - Web GUI Application

A Flask/Dash web interface for running the Automatic Status Tool.

Author: Moez Labiadh

Created: 2026-01-13
"""

import os, shutil, threading, uuid, pickle, re
from pathlib import Path
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from flask import Flask, send_file, render_template_string
from ast_processor import ASTProcessor

# Flask & Dash Setup
server = Flask(__name__)
server.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key')
server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

app = dash.Dash(__name__, server=server, 
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True)

job_store = {}

# Layout
app.layout = html.Div([
    dcc.Store(id='job-id'),
    dcc.Store(id='uploaded-files'),
    dcc.Interval(id='progress-interval', interval=1000, disabled=True),
    
    # Header
    dbc.Container([
        dbc.Row([dbc.Col([
            html.H1([html.I(className="fas fa-map-marked-alt me-3"), 
                "AST Lite - Automatic Status Tool"], className="text-white mb-0"),
            html.P("Spatial overlay analysis for AOI conflict detection", 
                className="text-white-50 mb-0")
        ])])
    ], fluid=True, className="bg-primary py-4 mb-4"),
    
    # Main Content
    dbc.Container([
        dbc.Row([
            # Left Column - Inputs
            dbc.Col([
                # Database Connections
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-database me-2"), "Database Connections"]),
                    dbc.CardBody([
                        dbc.Alert([html.I(className="fas fa-info-circle me-2"),
                            "Credentials are only used for this session."], color="info", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("BCGW/Oracle Connection", className="fw-bold"),
                                dbc.Input(id="bcgw-username", placeholder="Username *", type="text", className="mb-2"),
                                dbc.Input(id="bcgw-password", placeholder="Password *", type="password", className="mb-2"),
                                dbc.Input(id="bcgw-hostname", value="bcgw.bcgov/idwprod1.bcgov", type="text"),
                            ], md=6),
                            dbc.Col([
                                html.Label("PostGIS Connection", className="fw-bold"),
                                dbc.Input(id="postgis-host", value="localhost", type="text", className="mb-2"),
                                dbc.Input(id="postgis-database", value="ast_local_datasets", type="text", className="mb-2"),
                                dbc.Input(id="postgis-username", value="postgres", type="text", className="mb-2"),
                                dbc.Input(id="postgis-password", value="admin", type="password"),
                            ], md=6),
                        ])
                    ])
                ], className="mb-4"),
                
                # AOI Configuration
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-map me-2"), "AOI Configuration"]),
                    dbc.CardBody([
                        dbc.Row([dbc.Col([
                            html.Label("Region", className="fw-bold"),
                            dbc.Select(id="region", options=[
                                {"label": "Cariboo", "value": "cariboo"},
                                {"label": "Kootenay", "value": "kootenay"},
                                {"label": "Northeast", "value": "northeast"},
                                {"label": "Omineca", "value": "omineca"},
                                {"label": "Skeena", "value": "skeena"},
                                {"label": "South Coast", "value": "south_coast"},
                                {"label": "Thompson Okanagan", "value": "thompson_okanagan"},
                                {"label": "West Coast", "value": "west_coast"},
                            ], value="west_coast")
                        ])], className="mb-4"),
                        
                        dbc.RadioItems(id="input-source", options=[
                            {"label": "TANTALIS", "value": "TANTALIS"},
                            {"label": "Upload File", "value": "UPLOAD"}
                        ], value="TANTALIS", inline=True, className="mb-3"),
                        
                        html.Div(id="tantalis-inputs", children=[
                            dbc.Row([
                                dbc.Col([html.Label("File Number"), 
                                    dbc.Input(id="file-number", placeholder="5408057", type="text")], md=4),
                                dbc.Col([html.Label("Disposition ID"), 
                                    dbc.Input(id="disposition-id", placeholder="943829", type="number")], md=4),
                                dbc.Col([html.Label("Parcel ID"), 
                                    dbc.Input(id="parcel-id", placeholder="977043", type="number")], md=4),
                            ])
                        ]),
                        
                        html.Div(id="file-upload", children=[
                            dcc.Upload(id='upload-file', children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"), html.Br(),
                                "Drag and Drop or ", html.A('Select File'), html.Br(),
                                html.Small('Supported files: Zipped Shapefile or Geodatabases (*.zip), KML, KMZ')
                            ]), style={'width':'100%','height':'140px','borderWidth':'2px',
                                'borderStyle':'dashed','borderRadius':'10px','textAlign':'center',
                                'backgroundColor':'#f8f9fa','display':'flex','alignItems':'center',
                                'justifyContent':'center'}, multiple=False),
                            dcc.Loading(id="loading-upload", children=html.Div(id='upload-status'))
                        ], style={'display':'none'})
                    ])
                ], className="mb-4"),
                
                # Workspace
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-folder-open me-2"), "Output Configuration"]),
                    dbc.CardBody([
                        html.Label("Workspace Directory", className="fw-bold"),
                        dbc.Input(id="workspace", 
                            value=r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\outputs\APP")
                    ])
                ], className="mb-4"),
                
                # Controls
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dbc.Button([html.I(className="fas fa-play me-2"), "Run Analysis"],
                                id="run-button", color="success", size="lg", className="w-100")], md=8),
                            dbc.Col([dbc.Button([html.I(className="fas fa-stop me-2"), "Cancel"],
                                id="cancel-button", color="danger", size="lg", className="w-100", disabled=True)], md=4),
                        ])
                    ])
                ], className="mb-4"),
            ], md=12, lg=6),
            
            # Right Column - Progress & Results
            dbc.Col([
                # Progress
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-tasks me-2"), "Progress"]),
                    dbc.CardBody([
                        html.Div(id="progress-container", children=[
                            dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3"),
                            html.Div(id="progress-text", children="Ready to start", className="text-muted"),
                            html.Div(id="dataset-status", className="mt-3")
                        ])
                    ])
                ], className="mb-4"),
                
                # Results Preview
                dbc.Card([
                    dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "Results Preview"]),
                    dbc.CardBody([
                        html.Div(id="results-preview-container", children=[
                            dbc.Alert("Run an analysis to see results here.", color="info")
                        ])
                    ])
                ])
            ], md=12, lg=6),
        ])
    ], fluid=True),
    
    # Footer
    html.Footer([
        dbc.Container([html.P("AST Lite - GeoBC © 2026", 
            className="text-center text-muted mb-0 py-3")], fluid=True)
    ], className="mt-5 border-top")
])

# Callbacks
@app.callback(
    [Output("job-id", "data", allow_duplicate=True),
     Output("progress-text", "children", allow_duplicate=True)],
    Input("cancel-button", "n_clicks"), State("job-id", "data"),
    prevent_initial_call=True)
def cancel_analysis(n_clicks, job_id):
    if n_clicks and job_id and job_id in job_store:
        job_store[job_id]['status'] = 'cancelled'
        return job_id, "Cancelling..."
    return job_id, ""

@app.callback(
    [Output("tantalis-inputs", "style"), Output("file-upload", "style")],
    Input("input-source", "value"))
def toggle_input(source):
    return ({"display": "block"}, {"display": "none"}) if source == "TANTALIS" else \
           ({"display": "none"}, {"display": "block"})

@app.callback(
    [Output("upload-status", "children"), Output("uploaded-files", "data")],
    Input("upload-file", "contents"),
    State("upload-file", "filename"), State("workspace", "value"))
def handle_upload(contents, filename, workspace):
    if not contents:
        return "", None
    
    import zipfile, io, base64, fiona
    
    upload_id = str(uuid.uuid4())
    upload_dir = Path(workspace or '.') / 'uploads' / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        ext = filename.lower()
        
        if ext.endswith('.kml'):
            kml_path = upload_dir / filename
            kml_path.write_bytes(decoded)
            layers = fiona.listlayers(str(kml_path))
            if not layers:
                return dbc.Alert("No layers found in KML", color="danger"), None
            
            # Read KML using XML parser (doesn't require KML driver)
            import geopandas as gpd
            import xml.etree.ElementTree as ET
            from shapely.geometry import Polygon, Point, LineString
            
            try:
                print(f"Reading KML with XML parser: {kml_path}")
                
                # Parse KML as XML
                tree = ET.parse(str(kml_path))
                root = tree.getroot()
                
                # Handle KML namespace
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                if not root.tag.startswith('{'):
                    ns = {'kml': ''}
                
                geometries = []
                
                # Find all Placemarks
                placemarks = root.findall('.//kml:Placemark', ns) if ns['kml'] else root.findall('.//Placemark')
                for placemark in placemarks:
                    # Try Polygon
                    polygon = placemark.find('.//kml:Polygon' if ns['kml'] else './/Polygon', ns)
                    if polygon is not None:
                        outer = polygon.find('.//kml:outerBoundaryIs//kml:coordinates' if ns['kml'] else './/outerBoundaryIs//coordinates', ns)
                        if outer is not None and outer.text:
                            coords = []
                            for coord in outer.text.strip().split():
                                parts = coord.split(',')
                                if len(parts) >= 2:
                                    coords.append((float(parts[0]), float(parts[1])))
                            if len(coords) >= 3:
                                geometries.append(Polygon(coords))
                                continue
                    
                    # Try Point
                    point = placemark.find('.//kml:Point' if ns['kml'] else './/Point', ns)
                    if point is not None:
                        coords_elem = point.find('.//kml:coordinates' if ns['kml'] else './/coordinates', ns)
                        if coords_elem is not None and coords_elem.text:
                            parts = coords_elem.text.strip().split(',')
                            if len(parts) >= 2:
                                geometries.append(Point(float(parts[0]), float(parts[1])))
                                continue
                    
                    # Try LineString
                    line = placemark.find('.//kml:LineString' if ns['kml'] else './/LineString', ns)
                    if line is not None:
                        coords_elem = line.find('.//kml:coordinates' if ns['kml'] else './/coordinates', ns)
                        if coords_elem is not None and coords_elem.text:
                            coords = []
                            for coord in coords_elem.text.strip().split():
                                parts = coord.split(',')
                                if len(parts) >= 2:
                                    coords.append((float(parts[0]), float(parts[1])))
                            if len(coords) >= 2:
                                geometries.append(LineString(coords))
                
                if not geometries:
                    raise Exception("No geometries found")
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
                geom_type = gdf.geometry.geom_type.iloc[0]
                print(f"Found {len(geometries)} features, type: {geom_type}")
                
                # Calculate area for polygons
                if geom_type in ['Polygon', 'MultiPolygon']:
                    gdf_proj = gdf.to_crs(epsg=3005)
                    area_m2 = gdf_proj.geometry.area.sum()
                    area_ha = area_m2 / 10000
                    info_text = f"Layer: {layers[0]} | Type: {geom_type} | Area: {area_ha:.2f} ha"
                else:
                    info_text = f"Layer: {layers[0]} | Type: {geom_type}"
                
                print(f"Success: {info_text}")
                
            except Exception as e:
                print(f"KML read error: {type(e).__name__}: {str(e)}")
                info_text = f"Layer: {layers[0]} (could not read geometry info)"
            
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "KML file uploaded successfully!", html.Br(),
                html.Small(info_text, className="text-muted")
            ], color="success"), {
                "workspace": workspace, "upload_id": upload_id, "file_type": "kml",
                "file_path": str(kml_path), "layer_name": layers[0]}
        
        elif ext.endswith('.kmz'):
            with zipfile.ZipFile(io.BytesIO(decoded)) as z:
                z.extractall(upload_dir)
            kml_files = list(upload_dir.rglob('*.kml'))
            if len(kml_files) != 1:
                return dbc.Alert("KMZ must contain exactly 1 KML", color="warning"), None
            layers = fiona.listlayers(str(kml_files[0]))
            if not layers:
                return dbc.Alert("No layers found in KML", color="danger"), None
            
            # Read KML using XML parser (doesn't require KML driver)
            import geopandas as gpd
            import xml.etree.ElementTree as ET
            from shapely.geometry import Polygon, Point, LineString
            
            try:
                print(f"Reading KMZ with XML parser: {kml_files[0]}")
                
                # Parse KML as XML
                tree = ET.parse(str(kml_files[0]))
                root = tree.getroot()
                
                # Handle KML namespace
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                if not root.tag.startswith('{'):
                    ns = {'kml': ''}
                
                geometries = []
                
                # Find all Placemarks
                placemarks = root.findall('.//kml:Placemark', ns) if ns['kml'] else root.findall('.//Placemark')
                for placemark in placemarks:
                    # Try Polygon
                    polygon = placemark.find('.//kml:Polygon' if ns['kml'] else './/Polygon', ns)
                    if polygon is not None:
                        outer = polygon.find('.//kml:outerBoundaryIs//kml:coordinates' if ns['kml'] else './/outerBoundaryIs//coordinates', ns)
                        if outer is not None and outer.text:
                            coords = []
                            for coord in outer.text.strip().split():
                                parts = coord.split(',')
                                if len(parts) >= 2:
                                    coords.append((float(parts[0]), float(parts[1])))
                            if len(coords) >= 3:
                                geometries.append(Polygon(coords))
                                continue
                    
                    # Try Point
                    point = placemark.find('.//kml:Point' if ns['kml'] else './/Point', ns)
                    if point is not None:
                        coords_elem = point.find('.//kml:coordinates' if ns['kml'] else './/coordinates', ns)
                        if coords_elem is not None and coords_elem.text:
                            parts = coords_elem.text.strip().split(',')
                            if len(parts) >= 2:
                                geometries.append(Point(float(parts[0]), float(parts[1])))
                                continue
                    
                    # Try LineString
                    line = placemark.find('.//kml:LineString' if ns['kml'] else './/LineString', ns)
                    if line is not None:
                        coords_elem = line.find('.//kml:coordinates' if ns['kml'] else './/coordinates', ns)
                        if coords_elem is not None and coords_elem.text:
                            coords = []
                            for coord in coords_elem.text.strip().split():
                                parts = coord.split(',')
                                if len(parts) >= 2:
                                    coords.append((float(parts[0]), float(parts[1])))
                            if len(coords) >= 2:
                                geometries.append(LineString(coords))
                
                if not geometries:
                    raise Exception("No geometries found")
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
                geom_type = gdf.geometry.geom_type.iloc[0]
                print(f"Found {len(geometries)} features, type: {geom_type}")
                
                # Calculate area for polygons
                if geom_type in ['Polygon', 'MultiPolygon']:
                    gdf_proj = gdf.to_crs(epsg=3005)
                    area_m2 = gdf_proj.geometry.area.sum()
                    area_ha = area_m2 / 10000
                    info_text = f"Layer: {layers[0]} | Type: {geom_type} | Area: {area_ha:.2f} ha"
                else:
                    info_text = f"Layer: {layers[0]} | Type: {geom_type}"
                
                print(f"Success: {info_text}")
                
            except Exception as e:
                print(f"KMZ read error: {type(e).__name__}: {str(e)}")
                info_text = f"Layer: {layers[0]} (could not read geometry info)"
            
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "KMZ file uploaded successfully!", html.Br(),
                html.Small(info_text, className="text-muted")
            ], color="success"), {
                "workspace": workspace, "upload_id": upload_id, "file_type": "kmz",
                "file_path": str(kml_files[0]), "layer_name": layers[0]}
        
        elif ext.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(decoded)) as z:
                z.extractall(upload_dir)
            
            shp = [f for f in upload_dir.rglob('*.shp')]
            gdb = [f for f in upload_dir.rglob('*.gdb') if f.is_dir()]
            
            if shp:
                if len(shp) > 1:
                    return dbc.Alert("Multiple shapefiles found", color="warning"), None
                
                # Get shapefile info
                layer_name = shp[0].stem
                
                # Read geometry info using geopandas
                import geopandas as gpd
                try:
                    gdf = gpd.read_file(str(shp[0]))
                    
                    # Get geometry type
                    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else "Unknown"
                    
                    # Calculate area in hectares (if polygon)
                    if geom_type in ['Polygon', 'MultiPolygon']:
                        # Reproject to appropriate CRS for area calculation if needed
                        if gdf.crs and gdf.crs.is_geographic:
                            # Use BC Albers (EPSG:3005) for area calculation
                            gdf_proj = gdf.to_crs(epsg=3005)
                        else:
                            gdf_proj = gdf
                        
                        area_m2 = gdf_proj.geometry.area.sum()
                        area_ha = area_m2 / 10000  # Convert m² to hectares
                        
                        info_text = f"Layer: {layer_name} | Type: {geom_type} | Area: {area_ha:.2f} ha"
                    else:
                        info_text = f"Layer: {layer_name} | Type: {geom_type}"
                except Exception as e:
                    info_text = f"Layer: {layer_name} (could not read geometry info)"
                
                return dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    "Shapefile uploaded successfully!", html.Br(),
                    html.Small(info_text, className="text-muted")
                ], color="success"), {
                    "workspace": workspace, "file_type": "shapefile", "file_path": str(shp[0])}
            
            elif gdb:
                layers = fiona.listlayers(str(gdb[0]))
                if len(layers) != 1:
                    return dbc.Alert("GDB must have 1 feature class", color="warning"), None
                
                # Read GDB info using geopandas
                import geopandas as gpd
                try:
                    gdf = gpd.read_file(str(gdb[0]), layer=layers[0])
                    
                    # Get geometry type
                    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else "Unknown"
                    
                    # Calculate area in hectares (if polygon)
                    if geom_type in ['Polygon', 'MultiPolygon']:
                        # Reproject to appropriate CRS for area calculation if needed
                        if gdf.crs and gdf.crs.is_geographic:
                            # Use BC Albers (EPSG:3005) for area calculation
                            gdf_proj = gdf.to_crs(epsg=3005)
                        else:
                            gdf_proj = gdf
                        
                        area_m2 = gdf_proj.geometry.area.sum()
                        area_ha = area_m2 / 10000  # Convert m² to hectares
                        
                        info_text = f"Layer: {layers[0]} | Type: {geom_type} | Area: {area_ha:.2f} ha"
                    else:
                        info_text = f"Layer: {layers[0]} | Type: {geom_type}"
                except Exception as e:
                    info_text = f"Layer: {layers[0]} (could not read geometry info)"
                
                return dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    "Geodatabase uploaded successfully!", html.Br(),
                    html.Small(info_text, className="text-muted")
                ], color="success"), {
                    "workspace": workspace, "file_type": "gdb", 
                    "file_path": str(gdb[0] / layers[0]), "gdb_path": str(gdb[0]), "fc_name": layers[0]}
        
        return dbc.Alert("Unsupported format", color="danger"), None
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger"), None

@app.callback(
    [Output("job-id", "data"), Output("run-button", "disabled"),
     Output("cancel-button", "disabled"), Output("progress-interval", "disabled"),
     Output("progress-container", "children")],
    Input("run-button", "n_clicks"),
    [State("input-source", "value"), State("file-number", "value"),
     State("disposition-id", "value"), State("parcel-id", "value"),
     State("uploaded-files", "data"), State("region", "value"), State("workspace", "value"),
     State("bcgw-username", "value"), State("bcgw-password", "value"), State("bcgw-hostname", "value"),
     State("postgis-host", "value"), State("postgis-database", "value"),
     State("postgis-username", "value"), State("postgis-password", "value")],
    prevent_initial_call=True)
def start_analysis(n_clicks, input_source, file_number, disp_id, parcel_id, uploaded_files,
                   region, workspace, bcgw_user, bcgw_pwd, bcgw_host, 
                   pg_host, pg_db, pg_user, pg_pwd):
    if not n_clicks:
        return None, False, True, True, [
            dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3"),
            html.Div(id="progress-text", children="Ready"), html.Div(id="dataset-status")]
    
    errors = []
    if not bcgw_user: errors.append("BCGW Username required")
    if not bcgw_pwd: errors.append("BCGW Password required")
    if not pg_pwd: errors.append("PostGIS Password required")
    if input_source == "TANTALIS":
        if not file_number: errors.append("File Number required")
        if not disp_id: errors.append("Disposition ID required")
        if not parcel_id: errors.append("Parcel ID required")
    elif not uploaded_files:
        errors.append("Please upload a file")
    
    if errors:
        return None, False, True, True, [
            dbc.Progress(id="progress-bar", value=0, className="mb-3"),
            html.Div(id="progress-text", children="Fix errors"),
            html.Div(id="dataset-status"),
            dbc.Alert([html.Ul([html.Li(e) for e in errors])], color="danger")]
    
    job_id = str(uuid.uuid4())
    job_store[job_id] = {'status': 'starting', 'progress': 0, 'message': 'Initializing...',
        'results': None, 'error': None, 'workspace': workspace, 'start_time': datetime.now()}
    
    config = {
        'input_source': input_source, 'region': region, 'workspace': workspace,
        'workspace_xls': r'W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\input_spreadsheets',
        'bcgw': {'username': bcgw_user, 'password': bcgw_pwd, 'hostname': bcgw_host},
        'postgis': {'host': pg_host, 'database': pg_db, 'user': pg_user, 'password': pg_pwd}}
    
    if input_source == "TANTALIS":
        config['tantalis'] = {'file_number': file_number, 'disposition_id': int(disp_id), 
            'parcel_id': int(parcel_id)}
    else:
        config['aoi_file'] = uploaded_files['file_path']
    
    threading.Thread(target=run_ast_process, args=(job_id, config), daemon=True).start()
    
    return job_id, True, False, False, [
        dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3"),
        html.Div(id="progress-text", children="Starting..."), html.Div(id="dataset-status")]

@app.callback(
    [Output("progress-bar", "value"), Output("progress-text", "children"),
     Output("dataset-status", "children"), Output("results-preview-container", "children"),
     Output("run-button", "disabled", allow_duplicate=True),
     Output("cancel-button", "disabled", allow_duplicate=True),
     Output("progress-interval", "disabled", allow_duplicate=True)],
    Input("progress-interval", "n_intervals"), State("job-id", "data"),
    prevent_initial_call=True)
def update_progress(n, job_id):
    if not job_id or job_id not in job_store:
        return 0, "No job", "", "", False, True, True
    
    job = job_store[job_id]
    
    if job['status'] == 'cancelled':
        return job['progress'], "⊗ Cancelled", "", \
            dbc.Alert("Cancelled", color="warning"), False, True, True
    elif job['status'] == 'completed':
        # Format execution time for progress text
        progress_text_complete = html.Div([
            html.Div("✓ Complete", className="text-success fw-bold mb-2")
        ])
        
        # Create execution time info box
        exec_time_box = None
        if 'execution_time' in job:
            et = job['execution_time']
            time_parts = []
            if et['hours'] > 0:
                time_parts.append(f"{et['hours']}h")
            if et['minutes'] > 0:
                time_parts.append(f"{et['minutes']}m")
            if et['seconds'] > 0 or not time_parts:
                time_parts.append(f"{et['seconds']}s")
            exec_time_str = " ".join(time_parts)
            exec_time_box = dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                f"Execution time: {exec_time_str}"
            ], color="info", className="mt-2 mb-0")
        
        dataset_status_content = exec_time_box if exec_time_box else ""
        
        return 100, progress_text_complete, dataset_status_content, create_preview(job['results'], job_id, job), False, True, True
    elif job['status'] == 'error':
        return job['progress'], "✗ Error", "", \
            dbc.Alert([html.H5("Error"), html.P(str(job['error']))], color="danger"), False, True, True
    
    return job['progress'], job['message'], "", "", True, False, False

# Helper Functions
def create_preview(results, job_id, job=None):
    if not results:
        return dbc.Alert("No results", color="warning")
    
    total = sum(results.get('conflict_counts', {}).values())
    failed = results.get('failed_datasets', 0)
    
    # Summary cards with icons
    cards = dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-database fa-2x text-primary mb-2"),
            html.H3(results['total_datasets'], className="text-primary mb-1"),
            html.P("Datasets Analyzed", className="mb-0 small")
        ])], className="text-center")], md=4),
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-shield-alt fa-2x mb-2", 
                   style={"color": "#dc3545" if total > 0 else "#28a745"}),
            html.H3(total, className="text-danger mb-1" if total > 0 else "text-success mb-1"),
            html.P("Total Conflicts", className="mb-0 small")
        ])], className="text-center")], md=4),
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-exclamation-triangle fa-2x mb-2",
                   style={"color": "#ffc107" if failed > 0 else "#6c757d"}),
            html.H3(failed, className="text-warning mb-1" if failed > 0 else "text-muted mb-1"),
            html.P("Failed Datasets", className="mb-0 small")
        ])], className="text-center")], md=4),
    ], className="mb-3")
    
    # Action buttons
    buttons = dbc.Row([
        dbc.Col([dbc.Button([html.I(className="fas fa-external-link-alt me-2"), "Open Full Report"],
            color="primary", size="lg", className="w-100", href=f"/results/{job_id}",
            external_link=True, target="_blank")], md=6),
        dbc.Col([dbc.Button([html.I(className="fas fa-download me-2"), "Download Excel"],
            color="success", size="lg", className="w-100", href=f"/download/{job_id}")], md=6),
    ], className="mb-3")
    
    # Failed datasets details (if any)
    failed_details = None
    if failed > 0 and 'failed_details' in results:
        failed_items = []
        for f in results['failed_details']:
            failed_items.append(html.Li([
                html.Strong(f['item']), html.Br(),
                html.Small(f"Reason: {f['reason']}", className="text-muted")
            ], className="mb-2"))
        
        failed_details = dbc.Alert([
            html.H5([html.I(className="fas fa-exclamation-triangle me-2"), 
                "Failed Datasets Details"], className="alert-heading"),
            html.Hr(),
            html.Ul(failed_items, className="mb-0")
        ], color="warning", className="mb-3")
    
    # Conflicts by dataset and category table
    table = None
    if total > 0 and 'all_datasets' in results:
        # Build grouped table data
        all_datasets = results['all_datasets']
        
        # Group datasets by category
        from collections import OrderedDict
        grouped = OrderedDict()
        for ds in all_datasets:
            if ds['status'] == 'conflict':  # Only show datasets with conflicts
                cat = ds['category']
                if cat not in grouped:
                    grouped[cat] = []
                grouped[cat].append({
                    'dataset': ds['item'],
                    'count': ds['count']
                })
        
        # Function to get color based on conflict count (yellow -> orange -> red)
        def get_conflict_color(count):
            if count < 5:
                # Yellow shades
                return "#ffd700"  # Gold
            elif count < 8:
                # Yellow-orange transition
                return "#ffb700"  # Amber
            elif count < 10:
                # Orange shades
                return "#ff8c00"  # Dark orange
            elif count < 15:
                # Orange-red transition
                return "#ff6b35"  # Coral
            else:
                # Red shades
                return "#dc3545"  # Bootstrap danger red
        
        # Build HTML table manually for better styling
        table_rows = []
        for category, datasets in grouped.items():
            # Category header row
            table_rows.append(
                html.Tr([
                    html.Td(
                        html.Strong(category, className="text-primary"),
                        colSpan=2,
                        className="bg-light",
                        style={'fontSize': '1.1em', 'padding': '12px'}
                    )
                ])
            )
            # Dataset rows
            for ds in datasets:
                color = get_conflict_color(ds['count'])
                table_rows.append(
                    html.Tr([
                        html.Td(ds['dataset'], style={'paddingLeft': '30px'}),
                        html.Td(
                            html.Span(
                                ds['count'],
                                className="badge",
                                style={'backgroundColor': color, 'color': 'white'}
                            ),
                            style={'textAlign': 'center', 'width': '100px'}
                        )
                    ])
                )
        
        table = html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-list-ul me-2"),
                    "Summary of Conflicts"
                ], className="mb-1"),
                html.Small("Click on Open Full Report for details", className="text-muted")
            ], className="mb-3"),
            html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Dataset", style={'width': '70%'}),
                            html.Th("Number of Overlaps", style={'width': '30%', 'textAlign': 'center'})
                        ])
                    ], className="table-light"),
                    html.Tbody(table_rows)
                ], className="table table-sm table-hover")
            ], style={'maxHeight': '500px', 'overflowY': 'auto', 'border': '1px solid #dee2e6', 'borderRadius': '4px'})
        ])
    else:
        table = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            "No conflicts detected!"
        ], color="success", className="mt-3")
    
    components = [cards, buttons]
    if failed_details:
        components.append(failed_details)
    components.append(table)
    
    return html.Div(components)

def create_mini_map(job_id):
    # Removed - no longer using dash-leaflet
    return html.Div()

def run_ast_process(job_id, config):
    try:
        job_store[job_id].update({'status': 'running', 'message': 'Connecting...'})
        processor = ASTProcessor(config,
            progress_callback=lambda p, m: job_store[job_id].update({'progress': p, 'message': m}),
            cancellation_check=lambda: job_store.get(job_id, {}).get('status') == 'cancelled')
        
        results = processor.run()
        
        if job_store[job_id]['status'] == 'cancelled':
            return
        
        # Calculate execution time
        start_time = job_store[job_id].get('start_time', datetime.now())
        end_time = datetime.now()
        execution_time_seconds = (end_time - start_time).total_seconds()
        
        # Format execution time
        hours = int(execution_time_seconds // 3600)
        minutes = int((execution_time_seconds % 3600) // 60)
        seconds = int(execution_time_seconds % 60)
        
        if 'gdf_aoi' in results:
            gdf = results['gdf_aoi'].to_crs(4326)
            b = gdf.total_bounds
            job_store[job_id]['aoi_bounds'] = {
                'center': [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2],
                'bounds': [[b[1], b[0]], [b[3], b[2]]]}
        
        job_store[job_id].update({'status': 'completed', 'progress': 100, 
            'message': 'Complete', 'results': results, 'end_time': end_time,
            'execution_time': {'hours': hours, 'minutes': minutes, 'seconds': seconds}})
        
        # Save for full report
        p = Path(config['workspace']) / 'web_results'
        p.mkdir(exist_ok=True)
        with open(p / f'{job_id}_results.pkl', 'wb') as f:
            pickle.dump({'results': results, 'config': config, 
                'timestamp': datetime.now().isoformat()}, f)
    except Exception as e:
        if job_store.get(job_id, {}).get('status') != 'cancelled':
            job_store[job_id].update({'status': 'error', 'error': str(e)})

# Flask Routes
@server.route('/download/<job_id>')
def download(job_id):
    if job_id not in job_store or job_store[job_id]['status'] != 'completed':
        return "Not available", 404
    r = job_store[job_id]['results']
    if not r or 'output_file' not in r:
        return "File not found", 404
    return send_file(r['output_file'], as_attachment=True,
        download_name=f"AST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

@server.route('/map/<job_id>/<filename>')
def serve_map(job_id, filename):
    """Serve map HTML files for iframe embedding."""
    if job_id not in job_store or job_store[job_id]['status'] != 'completed':
        return "Not available", 404
    
    r = job_store[job_id]['results']
    workspace = r.get('workspace', '.')
    map_file = Path(workspace) / 'maps' / filename
    
    if not map_file.exists():
        return "Map not found", 404
    
    return send_file(map_file, mimetype='text/html')

@server.route('/results/<job_id>')
def full_report(job_id):
    if job_id not in job_store or job_store[job_id]['status'] != 'completed':
        return "Not available", 404
    r = job_store[job_id]['results']
    if not r:
        return "No data", 404
    
    ws = Path(job_store[job_id].get('workspace', '.'))
    pkl = ws / 'web_results' / f'{job_id}_results.pkl'
    det = None
    if pkl.exists():
        with open(pkl, 'rb') as f:
            det = pickle.load(f)
    
    # Pass job_store data for AOI bounds
    job_data = job_store[job_id]
    return render_template_string(generate_html(job_id, r, det, job_data))

def generate_html(job_id, res, det, job_data=None):
    total = sum(res.get('conflict_counts', {}).values())
    failed = res.get('failed_datasets', 0)
    workspace = res.get('workspace', '.')
    maps_dir = Path(workspace) / 'maps'
    
    # Get AOI bounds if available
    aoi_bounds = None
    if job_data and 'aoi_bounds' in job_data:
        aoi_bounds = job_data['aoi_bounds']
    
    # Build comprehensive table showing all datasets
    rows = ""
    
    # If we have the new all_datasets structure, use it
    if 'all_datasets' in res and res['all_datasets']:
        for dataset in res['all_datasets']:
            category = dataset.get('category', 'N/A')
            item = dataset.get('item', 'Unknown')
            status = dataset.get('status', 'unknown')
            
            if status == 'conflict' and item in res.get('conflict_details', {}):
                # Show each conflict as a separate row
                for conf in res['conflict_details'][item]:
                    details = conf.get('details', 'No details available')
                    rows += f'<tr><td>{category}</td><td>{item}</td><td>{details}</td></tr>'
            elif status == 'no_overlap':
                # Show "No overlaps" for datasets with no conflicts
                rows += f'<tr><td>{category}</td><td>{item}</td><td class="text-success"><em>No overlaps</em></td></tr>'
    else:
        # Fallback to old method (only conflicts)
        if 'conflict_details' in res and res['conflict_details']:
            for item, confs in res['conflict_details'].items():
                category = ""
                for c in confs:
                    if not category:
                        category = c.get("category", "N/A")
                    details = c.get("details", "")
                    rows += f'<tr><td>{category}</td><td>{item}</td><td>{details}</td></tr>'
    
    # Build map tabs for all generated maps
    map_html = ""
    # Look for the combined all-layers map
    all_layers_map = maps_dir / '00_all_layers.html'
    
    if all_layers_map.exists():
        # Display only the combined all-layers map
        relative_path = f'/map/{job_id}/00_all_layers.html'
        map_html = f'''
            <iframe src="{relative_path}" 
                    style="width: 100%; height: 800px; border: none;" 
                    allowfullscreen>
            </iframe>
        '''
    else:
        map_html = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>No map available. The combined layers map (00_all_layers.html) was not found.</div>'
    
    # Build failed datasets section (removed - not needed for display)
    failed_section = ""
    
    return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AST Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #f8f9fa; }}
        .map-container {{ border: 2px solid #dee2e6; border-radius: 8px; overflow: hidden; background: white; }}
        .table {{ background-color: white; }}
        .nav-tabs .nav-link {{ color: #495057; }}
        .nav-tabs .nav-link.active {{ color: #0d6efd; font-weight: bold; }}
        .tab-content {{ padding: 0; }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h2 class="mb-1"><i class="fas fa-map-marked-alt me-2"></i>AST Lite - Full Results Report</h2>
                <p class="mb-0">Interactive map and detailed conflicts table</p>
            </div>
            <div class="card-body">
                <p><strong>Job:</strong> {job_id} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <div class="mb-3">
                    <h5>Export Options</h5>
                    <a href="/download/{job_id}" class="btn btn-success me-2">
                        <i class="fas fa-download me-2"></i>Download Excel
                    </a>
                    <button onclick="window.print()" class="btn btn-secondary">
                        <i class="fas fa-print me-2"></i>Print
                    </button>
                </div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header bg-info text-white">
                <h4 class="mb-0"><i class="fas fa-globe me-2"></i>Interactive Map</h4>
                <small>Use layer controls to toggle datasets on/off. Click on features to see details.Check the output folder for individual maps</small>
            </div>
            <div class="card-body p-0">
                <div class="map-container">
                    {map_html}
                </div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header bg-dark text-white">
                <h4 class="mb-0"><i class="fas fa-table me-2"></i>Analysis Results - All Datasets</h4>
                <small>Showing all analyzed datasets with conflict details or "No overlaps" status</small>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Category</th>
                                <th>Dataset</th>
                                <th>Details / Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows if rows else '<tr><td colspan="3" class="text-center text-success">No conflicts detected</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        {failed_section}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))