"""
AST LITE - Web GUI Application

A Flask/Dash web interface for running the Automatic Status Tool.

Author:  Moez Labiadh - GeoBC

Created: 2026-01-06
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import threading
import uuid

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from flask import Flask, send_file, request
import plotly.graph_objects as go

# Import the AST processing functions
from ast_processor import ASTProcessor

# ============================================================================
# FLASK APP SETUP
# ============================================================================

server = Flask(__name__)
server.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
server.config['UPLOAD_FOLDER'] = 'uploads'
server.config['OUTPUT_FOLDER'] = 'outputs'
server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create necessary directories
os.makedirs(server.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(server.config['OUTPUT_FOLDER'], exist_ok=True)

# ============================================================================
# DASH APP SETUP
# ============================================================================

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Store for tracking job status
job_store = {}

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_header():
    """Create application header."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-map-marked-alt me-3"),
                    "AST Lite - Automatic Status Tool"
                ], className="text-white mb-0"),
                html.P(
                    "Spatial overlay analysis for AOI conflict detection",
                    className="text-white-50 mb-0"
                )
            ])
        ])
    ], fluid=True, className="bg-primary py-4 mb-4")


def create_connection_card():
    """Create database connection configuration card."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-database me-2"),
            "Database Connections"
        ]),
        dbc.CardBody([
            dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    "Your credentials are only used for this session and are never stored."
                ],
                color="info",
                className="mb-3"
            ),
            dbc.Row([
                dbc.Col([
                    html.Label("BCGW/Oracle Connection", className="fw-bold"),
                    dbc.Input(
                        id="bcgw-username",
                        placeholder="Username *",
                        type="text",
                        className="mb-2",
                        required=True
                    ),
                    dbc.Input(
                        id="bcgw-password",
                        placeholder="Password *",
                        type="password",
                        className="mb-2",
                        required=True
                    ),
                    dbc.Input(
                        id="bcgw-hostname",
                        placeholder="Hostname",
                        value="bcgw.bcgov/idwprod1.bcgov",
                        type="text"
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("PostGIS Connection", className="fw-bold"),
                    dbc.Input(
                        id="postgis-host",
                        placeholder="Host",
                        value="localhost",
                        type="text",
                        className="mb-2"
                    ),
                    dbc.Input(
                        id="postgis-database",
                        placeholder="Database",
                        value="ast_local_datasets",
                        type="text",
                        className="mb-2"
                    ),
                    dbc.Input(
                        id="postgis-username",
                        placeholder="Username",
                        value="postgres",
                        type="text",
                        className="mb-2"
                    ),
                    dbc.Input(
                        id="postgis-password",
                        placeholder="Password *",
                        value="admin",
                        type="password",
                        required=True
                    ),
                ], md=6),
            ]),
            html.Small("* Required fields", className="text-muted")
        ])
    ], className="mb-4")


def create_input_card():
    """Create AOI input configuration card."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-map me-2"),
            "Area of Interest (AOI) Configuration"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Input Source", className="fw-bold"),
                    dbc.RadioItems(
                        id="input-source",
                        options=[
                            {"label": "TANTALIS", "value": "TANTALIS"},
                            {"label": "Upload Shapefile", "value": "AOI"}
                        ],
                        value="TANTALIS",
                        inline=True
                    )
                ], md=12, className="mb-3"),
            ]),
            
            # TANTALIS inputs (shown by default)
            html.Div(id="tantalis-inputs", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("File Number"),
                        dbc.Input(id="file-number", placeholder="e.g., 5408057", type="text")
                    ], md=4),
                    dbc.Col([
                        html.Label("Disposition ID"),
                        dbc.Input(id="disposition-id", placeholder="e.g., 943829", type="number")
                    ], md=4),
                    dbc.Col([
                        html.Label("Parcel ID"),
                        dbc.Input(id="parcel-id", placeholder="e.g., 977043", type="number")
                    ], md=4),
                ])
            ]),
            
            # AOI file upload (hidden by default)
            html.Div(id="aoi-upload", children=[
                dcc.Upload(
                    id='upload-shapefile',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                        html.Br(),
                        html.Span('Drag and Drop or ', style={'lineHeight': 'normal'}),
                        html.A('Select Shapefile (.shp, .shx, .dbf, .prj)', style={'lineHeight': 'normal'})
                    ], style={'lineHeight': 'normal', 'padding': '20px'}),
                    style={
                        'width': '100%',
                        'height': '120px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'textAlign': 'center',
                        'backgroundColor': '#f8f9fa',
                        'marginBottom': '15px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    },
                    multiple=True
                ),
                html.Div(id='upload-status', className="mt-2")
            ], style={'display': 'none'}),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Region", className="fw-bold mt-3"),
                    dbc.Select(
                        id="region",
                        options=[
                            {"label": "Cariboo", "value": "cariboo"},
                            {"label": "Kootenay", "value": "kootenay"},
                            {"label": "Northeast", "value": "northeast"},
                            {"label": "Omineca", "value": "omineca"},
                            {"label": "Skeena", "value": "skeena"},
                            {"label": "South Coast", "value": "south_coast"},
                            {"label": "Thompson Okanagan", "value": "thompson_okanagan"},
                            {"label": "West Coast", "value": "west_coast"},
                        ],
                        value="cariboo"
                    )
                ], md=6),
                dbc.Col([
                    html.Label("Workspace/Output Directory", className="fw-bold mt-3"),
                    dbc.Input(
                        id="workspace",
                        placeholder="Path to workspace directory",
                        value=r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\outputs\APP",
                        type="text"
                    )
                ], md=6),
            ])
        ])
    ], className="mb-4")


def create_control_card():
    """Create process control card."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-play me-2"), "Run Analysis"],
                        id="run-button",
                        color="success",
                        size="lg",
                        className="w-100"
                    )
                ], md=8),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-stop me-2"), "Cancel"],
                        id="cancel-button",
                        color="danger",
                        size="lg",
                        className="w-100",
                        disabled=True
                    )
                ], md=4),
            ])
        ])
    ], className="mb-4")


def create_progress_card():
    """Create progress monitoring card."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-tasks me-2"),
            "Progress"
        ]),
        dbc.CardBody([
            html.Div(id="progress-container", children=[
                dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3"),
                html.Div(id="progress-text", children="Ready to start", className="text-muted"),
                html.Div(id="dataset-status", className="mt-3")
            ])
        ])
    ], className="mb-4")


def create_results_card():
    """Create results display card."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-bar me-2"),
            "Results"
        ]),
        dbc.CardBody([
            html.Div(id="results-container", children=[
                dbc.Alert(
                    "Run an analysis to see results here.",
                    color="info",
                    className="mb-0"
                )
            ])
        ])
    ])


# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = html.Div([
    dcc.Store(id='job-id'),
    dcc.Store(id='uploaded-files'),
    dcc.Interval(id='progress-interval', interval=1000, disabled=True),
    
    create_header(),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                create_connection_card(),
                create_input_card(),
                create_control_card(),
            ], md=12, lg=6),
            dbc.Col([
                create_progress_card(),
                create_results_card(),
            ], md=12, lg=6),
        ])
    ], fluid=True),
    
    html.Footer([
        dbc.Container([
            html.P(
                "AST Lite - GeoBC © 2026",
                className="text-center text-muted mb-0 py-3"
            )
        ], fluid=True)
    ], className="mt-5 border-top")
])


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output("job-id", "data", allow_duplicate=True),
     Output("progress-text", "children", allow_duplicate=True)],
    Input("cancel-button", "n_clicks"),
    State("job-id", "data"),
    prevent_initial_call=True
)
def cancel_analysis(n_clicks, job_id):
    """Cancel the running analysis."""
    if not n_clicks or not job_id:
        return job_id, ""
    
    if job_id in job_store:
        job_store[job_id]['status'] = 'cancelled'
        job_store[job_id]['message'] = 'Cancelling analysis...'
        return job_id, "Cancelling analysis..."
    
    return job_id, ""


@app.callback(
    [Output("tantalis-inputs", "style"),
     Output("aoi-upload", "style")],
    Input("input-source", "value")
)
def toggle_input_source(input_source):
    """Toggle between TANTALIS and AOI upload inputs."""
    if input_source == "TANTALIS":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}


@app.callback(
    [Output("upload-status", "children"),
     Output("uploaded-files", "data")],
    Input("upload-shapefile", "contents"),
    State("upload-shapefile", "filename")
)
def handle_file_upload(contents, filenames):
    """Handle shapefile upload."""
    if not contents:
        return "", None
    
    upload_id = str(uuid.uuid4())
    upload_dir = Path(server.config['UPLOAD_FOLDER']) / upload_id
    upload_dir.mkdir(exist_ok=True)
    
    uploaded = []
    for content, filename in zip(contents, filenames):
        # Decode and save file
        content_type, content_string = content.split(',')
        import base64
        decoded = base64.b64decode(content_string)
        
        file_path = upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(decoded)
        uploaded.append(filename)
    
    # Find the .shp file
    shp_file = next((f for f in uploaded if f.endswith('.shp')), None)
    
    if not shp_file:
        return dbc.Alert("Please upload a .shp file", color="danger"), None
    
    status = dbc.Alert([
        html.I(className="fas fa-check-circle me-2"),
        f"Uploaded: {', '.join(uploaded)}"
    ], color="success")
    
    return status, {"upload_id": upload_id, "shp_file": shp_file}


@app.callback(
    [Output("job-id", "data"),
     Output("run-button", "disabled"),
     Output("cancel-button", "disabled"),
     Output("progress-interval", "disabled"),
     Output("progress-container", "children")],
    Input("run-button", "n_clicks"),
    [State("input-source", "value"),
     State("file-number", "value"),
     State("disposition-id", "value"),
     State("parcel-id", "value"),
     State("uploaded-files", "data"),
     State("region", "value"),
     State("workspace", "value"),
     State("bcgw-username", "value"),
     State("bcgw-password", "value"),
     State("bcgw-hostname", "value"),
     State("postgis-host", "value"),
     State("postgis-database", "value"),
     State("postgis-username", "value"),
     State("postgis-password", "value")],
    prevent_initial_call=True
)
def start_analysis(n_clicks, input_source, file_number, disp_id, parcel_id,
                   uploaded_files, region, workspace, bcgw_user, bcgw_pwd,
                   bcgw_host, pg_host, pg_db, pg_user, pg_pwd):
    """Start the AST analysis in a background thread."""
    if not n_clicks:
        initial_progress = [
            dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3"),
            html.Div(id="progress-text", children="Ready to start", className="text-muted"),
            html.Div(id="dataset-status", className="mt-3")
        ]
        return None, False, True, True, initial_progress
    
    # Validate credentials
    validation_errors = []
    
    if not bcgw_user or not bcgw_user.strip():
        validation_errors.append("BCGW Username is required")
    if not bcgw_pwd or not bcgw_pwd.strip():
        validation_errors.append("BCGW Password is required")
    if not pg_pwd or not pg_pwd.strip():
        validation_errors.append("PostGIS Password is required")
    
    # Validate input-specific requirements
    if input_source == "TANTALIS":
        if not file_number or not file_number.strip():
            validation_errors.append("File Number is required")
        if not disp_id:
            validation_errors.append("Disposition ID is required")
        if not parcel_id:
            validation_errors.append("Parcel ID is required")
    else:
        if not uploaded_files:
            validation_errors.append("Please upload a shapefile")
    
    # Show validation errors
    if validation_errors:
        error_alert = dbc.Alert([
            html.H5("Please fix the following errors:", className="alert-heading"),
            html.Ul([html.Li(error) for error in validation_errors])
        ], color="danger", className="mb-3")
        
        initial_progress = [
            dbc.Progress(id="progress-bar", value=0, striped=True, className="mb-3"),
            html.Div(id="progress-text", children="Please provide required information", className="text-muted"),
            html.Div(id="dataset-status", className="mt-3"),
            error_alert
        ]
        return None, False, True, True, initial_progress
    
    # Create job
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing...',
        'current_dataset': '',
        'total_datasets': 0,
        'completed_datasets': 0,
        'results': None,
        'error': None
    }
    
    # Prepare configuration
    config = {
        'input_source': input_source,
        'region': region,
        'workspace': workspace,
        'workspace_xls': r'W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\input_spreadsheets',
        'bcgw': {
            'username': bcgw_user,
            'password': bcgw_pwd,
            'hostname': bcgw_host
        },
        'postgis': {
            'host': pg_host,
            'database': pg_db,
            'user': pg_user,
            'password': pg_pwd
        }
    }
    
    if input_source == "TANTALIS":
        config['tantalis'] = {
            'file_number': file_number,
            'disposition_id': int(disp_id),
            'parcel_id': int(parcel_id)
        }
    else:
        upload_dir = Path(server.config['UPLOAD_FOLDER']) / uploaded_files['upload_id']
        config['aoi_file'] = str(upload_dir / uploaded_files['shp_file'])
    
    # Start processing in background thread
    thread = threading.Thread(
        target=run_ast_process,
        args=(job_id, config),
        daemon=True
    )
    thread.start()
    
    success_progress = [
        dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="mb-3"),
        html.Div(id="progress-text", children="Starting analysis...", className="text-muted"),
        html.Div(id="dataset-status", className="mt-3")
    ]
    
    return job_id, True, False, False, success_progress


@app.callback(
    [Output("progress-bar", "value"),
     Output("progress-text", "children"),
     Output("dataset-status", "children"),
     Output("results-container", "children"),
     Output("run-button", "disabled", allow_duplicate=True),
     Output("cancel-button", "disabled", allow_duplicate=True),
     Output("progress-interval", "disabled", allow_duplicate=True)],
    Input("progress-interval", "n_intervals"),
    State("job-id", "data"),
    prevent_initial_call=True
)
def update_progress(n_intervals, job_id):
    """Update progress display."""
    if not job_id or job_id not in job_store:
        return 0, "No active job", "", "", False, True, True
    
    job = job_store[job_id]
    
    progress_val = job['progress']
    progress_text = job['message']
    
    # Check if job was cancelled
    if job['status'] == 'cancelled':
        results = dbc.Alert([
            html.H5("Analysis Cancelled", className="alert-heading"),
            html.P("The analysis was stopped by user request.")
        ], color="warning")
        return progress_val, "⊗ Cancelled", "", results, False, True, True
    
    # Results
    if job['status'] == 'completed':
        results = create_results_display(job['results'], job_id)
        # Re-enable the Run button, disable Cancel, stop progress updates
        return 100, "✓ Analysis Complete", "", results, False, True, True
    elif job['status'] == 'error':
        results = dbc.Alert([
            html.H5("Error", className="alert-heading"),
            html.P(str(job['error'])),
            html.Hr(),
            html.P("Please check your inputs and try again.", className="mb-0")
        ], color="danger")
        # Re-enable the Run button, disable Cancel, stop progress updates
        return progress_val, "✗ Error", "", results, False, True, True
    else:
        # Still running - keep buttons in running state, no extra dataset status
        return progress_val, progress_text, "", "", True, False, False


def create_results_display(results, job_id):
    """Create results display component."""
    if not results:
        return dbc.Alert("No results available", color="warning")
    
    # Summary statistics
    total_conflicts = sum(results.get('conflict_counts', {}).values())
    failed_count = results.get('failed_datasets', 0)
    
    summary_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(results['total_datasets'], className="text-primary"),
                    html.P("Datasets Analyzed", className="mb-0")
                ])
            ])
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(total_conflicts, className="text-danger"),
                    html.P("Total Conflicts", className="mb-0")
                ])
            ])
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(failed_count, className="text-warning"),
                    html.P("Failed Datasets", className="mb-0")
                ])
            ])
        ], md=4),
    ], className="mb-4")
    
    # Download button
    download_section = dbc.Row([
        dbc.Col([
            dbc.Button(
                [html.I(className="fas fa-download me-2"), "Download Excel Report"],
                id="download-button",
                color="primary",
                href=f"/download/{job_id}",
                external_link=True,
                className="w-100"
            )
        ])
    ], className="mb-4")
    
    # Failed datasets details
    failed_details = None
    if failed_count > 0 and 'failed_details' in results:
        failed_items = []
        for failed in results['failed_details']:
            failed_items.append(
                html.Li([
                    html.Strong(failed['item']),
                    html.Br(),
                    html.Small(f"Reason: {failed['reason']}", className="text-muted")
                ], className="mb-2")
            )
        
        failed_details = dbc.Alert([
            html.H5([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Failed Datasets Details"
            ], className="alert-heading"),
            html.Hr(),
            html.Ul(failed_items, className="mb-0")
        ], color="warning", className="mb-4")
    
    # Conflict details
    if total_conflicts > 0 and 'conflicts_by_category' in results:
        conflict_details = html.Div([
            html.H5("Conflicts by Category", className="mb-3"),
            dash_table.DataTable(
                data=results['conflicts_by_category'],
                columns=[
                    {"name": "Category", "id": "category"},
                    {"name": "Count", "id": "count"}
                ],
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'}
            )
        ])
    else:
        conflict_details = dbc.Alert("No conflicts detected", color="success")
    
    components = [summary_cards, download_section]
    if failed_details:
        components.append(failed_details)
    components.append(conflict_details)
    
    return html.Div(components)


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def run_ast_process(job_id, config):
    """Run the AST process in background thread."""
    try:
        # Update status
        job_store[job_id]['status'] = 'running'
        job_store[job_id]['message'] = 'Connecting to databases...'
        
        # Initialize processor with cancellation check
        processor = ASTProcessor(
            config,
            progress_callback=lambda p, m: update_job_progress(job_id, p, m),
            cancellation_check=lambda: job_store.get(job_id, {}).get('status') == 'cancelled'
        )
        
        # Run analysis
        results = processor.run()
        
        # Check if cancelled during processing
        if job_store[job_id]['status'] == 'cancelled':
            job_store[job_id]['message'] = 'Analysis cancelled by user'
            return
        
        # Store results
        job_store[job_id]['status'] = 'completed'
        job_store[job_id]['progress'] = 100
        job_store[job_id]['message'] = 'Analysis complete'
        job_store[job_id]['results'] = results
        
    except Exception as e:
        # Check if error is due to cancellation
        if job_store.get(job_id, {}).get('status') == 'cancelled':
            job_store[job_id]['message'] = 'Analysis cancelled by user'
        else:
            job_store[job_id]['status'] = 'error'
            job_store[job_id]['error'] = str(e)
            print(f"Error in job {job_id}: {e}")


def update_job_progress(job_id, progress, message):
    """Update job progress from background thread."""
    if job_id in job_store:
        job_store[job_id]['progress'] = progress
        job_store[job_id]['message'] = message


# ============================================================================
# FLASK ROUTES
# ============================================================================

@server.route('/download/<job_id>')
def download_results(job_id):
    """Download Excel results file."""
    if job_id not in job_store or job_store[job_id]['status'] != 'completed':
        return "Results not available", 404
    
    results = job_store[job_id]['results']
    if not results or 'output_file' not in results:
        return "Output file not found", 404
    
    return send_file(
        results['output_file'],
        as_attachment=True,
        download_name=f"AST_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # For local development
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8050))
    )