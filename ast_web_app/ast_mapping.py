"""
ast_mapping.py

This module provides interactive HTML map generation for the AST Lite tool.

Author: Moez Labiadh - GeoBC
Created: 2025-01-19
"""

import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MeasureControl, MousePosition, MiniMap, GroupedLayerControl
from branca.element import Template, MacroElement


# ============================================================================
# CSS STYLING
# ============================================================================

MAP_CSS = """
{% macro html(this, kwargs) %}
    <style>
/* Marker PopUp Box CSS */
        .leaflet-popup-content-wrapper{
            padding: 1px;
            text-align: left;
            border: 1px solid #d7a45d;
            border-radius: 12px;
        }
        .leaflet-popup-content{
            margin: 13px 24px 13px 20px;
            font-size: 1.2em;
            line-height: 1.3;
            min-height: 1px;
        }

/* Layer Control Panel CSS */
        .leaflet-control-layers-list {
            width: 16vw;
            max-height: 350px;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .leaflet-control-layers form {
            z-index: 10000;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .leaflet-control-layers-group-label{
            padding: 2px;
            margin: 2px;
            background-color: #e09494;
            border: 1px dashed black;
            border-radius: 4px;
            text-align: center;
        }
    </style>
{% endmacro %}
"""


# ============================================================================
# MAP GENERATOR CLASS
# ============================================================================

class MapGenerator:
    """
    Enhanced map generator for AST Lite.
    
    Creates interactive HTML maps with:
    - Multiple basemap options (GeoBC, satellite)
    - AOI with configurable buffers
    - Layer legends with dynamic colors
    - Grouped layer controls
    - BC Government branding
    """
    
    # Default color palette for layers
    COLOR_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    def __init__(
        self,
        gdf_aoi: gpd.GeoDataFrame,
        workspace: str,
        df_stat: pd.DataFrame,
        logo_path: Optional[str] = None
    ):
        """
        Initialize the MapGenerator.
        
        Args:
            gdf_aoi: GeoDataFrame containing the Area of Interest
            workspace: Output workspace directory
            df_stat: Configuration DataFrame with buffer distances
            logo_path: Optional path to logo image for branding
        """
        self.gdf_aoi = gdf_aoi
        self.workspace = workspace
        self.df_stat = df_stat
        self.logo_path = logo_path
        
        # Ensure AOI is in WGS84 for mapping
        self.gdf_aoi_4326 = gdf_aoi.to_crs(epsg=4326)
        
        # Create maps output directory
        self.maps_dir = os.path.join(workspace, 'maps')
        os.makedirs(self.maps_dir, exist_ok=True)
        
        # Generate buffers based on config
        self.buffer_gdfs = self._create_buffers()
        
        # Get map bounds (use largest buffer or AOI)
        self.bounds = self._get_map_bounds()
        
        # Initialize all-layers map
        self.map_all = None
        self.all_layer_groups = []  # Track layer groups for all-layers map
        self.category_groups = {}  # Track groups by category
        
        # Track AOI groups for grouped layer control
        self.aoi_groups_all = []
        
        # Track layer colors for all-layers legend
        self.all_layers_legend = []  # List of (color, layer_name) tuples
        
        # Color index for assigning unique colors to each layer
        self._color_index = 0
    
    def _create_buffers(self) -> Dict[int, gpd.GeoDataFrame]:
        """
        Create buffer GeoDataFrames based on unique buffer distances in config.
        
        Returns:
            Dictionary mapping buffer distance to GeoDataFrame
        """
        # Extract unique buffer distances from config
        buffer_distances = (
            pd.to_numeric(
                self.df_stat['Buffer_Distance'].astype(str).str.strip(), 
                errors='coerce'
            )
            .dropna()
            .astype(int)
            .unique()
        )
        
        # Always include 1000m buffer as default
        if 1000 not in buffer_distances:
            buffer_distances = np.append(buffer_distances, 1000)
        
        buffer_distances = np.sort(buffer_distances)
        
        # Create buffer GeoDataFrames
        buffer_gdfs = {}
        for distance in buffer_distances:
            if distance > 0:
                print(f'....Creating {distance}m buffer for mapping')
                buffer_gdfs[distance] = gpd.GeoDataFrame(
                    geometry=self.gdf_aoi.buffer(distance),
                    crs=self.gdf_aoi.crs
                )
        
        return buffer_gdfs
    
    def _get_map_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get map bounds based on largest buffer or AOI.
        
        Returns:
            Tuple of (ymin, xmin, ymax, xmax) in WGS84
        """
        if self.buffer_gdfs:
            # Use largest buffer for bounds
            max_distance = max(self.buffer_gdfs.keys())
            bounds_gdf = self.buffer_gdfs[max_distance].to_crs(epsg=4326)
        else:
            bounds_gdf = self.gdf_aoi_4326
        
        xmin, ymin, xmax, ymax = bounds_gdf.total_bounds
        return (ymin, xmin, ymax, xmax)
    
    def _create_map_template(
        self,
        fit_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> folium.Map:
        """
        Create a base folium map with standard configuration.
        
        Args:
            fit_bounds: Optional tuple of (ymin, xmin, ymax, xmax) to fit map to
        
        Returns:
            Configured folium Map object
        """
        # Create map object
        map_obj = folium.Map(control_scale=True, tiles=None)
        
        # Use provided bounds or default
        bounds = fit_bounds if fit_bounds else self.bounds
        ymin, xmin, ymax, xmax = bounds
        
        # Fit map to bounds
        map_obj.fit_bounds([[ymin, xmin], [ymax, xmax]])
        
        # Add GeoBC basemap
        geobc_url = 'https://maps.gov.bc.ca/arcgis/rest/services/province/web_mercator_cache/MapServer/tile/{z}/{y}/{x}'
        geobc_attr = 'GeoBC, DataBC, TomTom, © OpenStreetMap Contributors'
        folium.TileLayer(
            tiles=geobc_url,
            name='GeoBC Basemap',
            attr=geobc_attr,
            overlay=False,
            control=True,
            show=True
        ).add_to(map_obj)
        
        # Add satellite basemap
        satellite_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        satellite_attr = 'Tiles &copy; Esri'
        folium.TileLayer(
            tiles=satellite_url,
            name='Imagery Basemap',
            attr=satellite_attr,
            overlay=False,
            control=True,
            show=False
        ).add_to(map_obj)
        
        # Add refresh view button
        map_var_name = map_obj.get_name()
        refresh_html = f"""
        <div style="position: fixed; 
            top: 88px; left: 70px; width: 150px; height: 70px; 
            background-color:transparent; border:0px solid grey; z-index: 900;">
            <button style="font-weight:bold; color:#DE1610; cursor:pointer;
                    padding: 5px 10px; border-radius: 4px; border: 1px solid #ccc;
                    background-color: white;"
                    onclick="{map_var_name}.fitBounds([[{ymin}, {xmin}], [{ymax}, {xmax}]])">
                Refresh View
            </button>
        </div>
        """
        map_obj.get_root().html.add_child(folium.Element(refresh_html))
        
        # Add measure controls
        map_obj.add_child(MeasureControl(
            primary_length_unit='meters',
            secondary_length_unit='kilometers',
            primary_area_unit='hectares'
        ))
        
        # Add mouse position
        MousePosition(
            position='bottomright',
            separator=' | ',
            prefix='Coordinates:',
            num_digits=5
        ).add_to(map_obj)
        
        # Add lat/long popup on click
        map_obj.add_child(folium.features.LatLngPopup())
        
        # Add minimap
        minimap = MiniMap(position="bottomleft", toggle_display=True)
        map_obj.add_child(minimap)
        
        # Offset minimap to avoid footer
        minimap_css = """
        <style>
            .leaflet-bottom.leaflet-left {
                margin-bottom: 80px;
            }
        </style>
        """
        map_obj.get_root().html.add_child(folium.Element(minimap_css))
        
        # Add custom CSS styling
        style = MacroElement()
        style._template = Template(MAP_CSS)
        map_obj.get_root().add_child(style)
        
        return map_obj
    
    def _add_aoi_layers(
        self,
        map_obj: folium.Map,
        show_buffers: bool = True
    ) -> List[folium.FeatureGroup]:
        """
        Add AOI and buffer layers to a map.
        
        Args:
            map_obj: Folium map object
            show_buffers: Whether to add buffer layers
        
        Returns:
            List of FeatureGroup objects for grouped layer control
        """
        aoi_groups = []
        
        # Add AOI layer
        grp_aoi = folium.FeatureGroup(name='AOI', show=True)
        folium.GeoJson(
            data=self.gdf_aoi_4326,
            name='AOI',
            style_function=lambda x: {
                'color': 'red',
                'fillColor': 'none',
                'weight': 3
            }
        ).add_to(grp_aoi)
        grp_aoi.add_to(map_obj)
        aoi_groups.append(grp_aoi)
        
        # Add buffer layers
        if show_buffers:
            for distance, gdf_buffer in sorted(self.buffer_gdfs.items()):
                grp_buffer = folium.FeatureGroup(
                    name=f'AOI_{distance} m',
                    show=False
                )
                folium.GeoJson(
                    data=gdf_buffer.to_crs(epsg=4326),
                    name=f'AOI_{distance}m',
                    style_function=lambda x: {
                        'color': 'orange',
                        'fillColor': 'none',
                        'weight': 2,
                        'dashArray': '5, 5'
                    }
                ).add_to(grp_buffer)
                grp_buffer.add_to(map_obj)
                aoi_groups.append(grp_buffer)
        
        return aoi_groups
    
    def _generate_legend_html(
        self,
        legend_items: List[Tuple[str, str]],
        header: str = "Legend",
        include_aoi: bool = True
    ) -> str:
        """
        Generate HTML for map legend.
        
        Args:
            legend_items: List of (color, label) tuples
            header: Legend header text
            include_aoi: Whether to include AOI/buffer entries
        
        Returns:
            HTML string for legend
        """
        legend_html = f'''
        <div id="legend" style="position: fixed; 
            bottom: 60px; right: 30px; z-index: 1000; 
            background-color: #fff; padding: 10px; 
            border-radius: 5px; border: 1px solid grey;
            max-height: 400px; overflow-y: auto;
            font-family: 'BC Sans', Arial, sans-serif;">
            <div style="font-weight: bold; margin-bottom: 8px; 
                        border-bottom: 1px solid #ccc; padding-bottom: 5px;">
                {header}
            </div>
        '''
        
        # Add layer items
        for color, label in legend_items:
            legend_html += f'''
            <div style="margin: 4px 0;">
                <span style="display: inline-block; 
                    margin-right: 8px; background-color: {color}; 
                    width: 15px; height: 15px; vertical-align: middle;"></span>
                <span style="vertical-align: middle;">{label}</span>
            </div>
            '''
        
        # Add AOI and buffer entries
        if include_aoi:
            legend_html += '''
            <div style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 8px;">
                <div style="margin: 4px 0;">
                    <span style="display: inline-block; 
                        margin-right: 8px; background-color: transparent;
                        border: 2px solid red;
                        width: 15px; height: 15px; vertical-align: middle;"></span>
                    <span style="vertical-align: middle;">AOI</span>
                </div>
                <div style="margin: 4px 0;">
                    <span style="display: inline-block; 
                        margin-right: 8px; background-color: transparent;
                        border: 2px dashed orange;
                        width: 15px; height: 15px; vertical-align: middle;"></span>
                    <span style="vertical-align: middle;">AOI Buffers</span>
                </div>
            </div>
            '''
        
        legend_html += '</div>'
        return legend_html
    
    def _inject_branding(
        self,
        map_path: str,
        title: str = "Feature Layer"
    ) -> None:
        """
        Inject BC Government branding (header/footer) into saved HTML map.
        
        Args:
            map_path: Path to the saved HTML map file
            title: Title text to display in header
        """
        map_path = Path(map_path)
        if not map_path.exists():
            print(f"Warning: Map file not found for branding: {map_path}")
            return
        
        # Read original HTML
        with map_path.open('r', encoding='utf-8') as f:
            html = f.read()
        
        # Add BC Sans font
        bc_sans_link = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/bcgov/bc-sans@main/css/BCSans.css" />'
        html = html.replace('<head>', f'<head>\n    {bc_sans_link}')
        
        # Prepare logo (base64 encoded if provided)
        logo_html = ""
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                with open(self.logo_path, "rb") as img_file:
                    logo_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Logo" style="height:50px;margin-right:15px;">'
            except Exception as e:
                print(f"Warning: Could not load logo: {e}")
        
        # Define header HTML
        header_html = f"""
        <div style="background:#003366; color:white; padding:10px 20px;
                    display:flex; align-items:center; font-family:'BC Sans', sans-serif;">
            {logo_html}
            <div style="display:flex; flex-direction:column;">
                <div style="font-size:16px; font-weight:bold; color:white;">
                    Water, Land, and Resource Stewardship
                </div>
                <div style="font-size:20px; font-weight:bold; color:#e3a82b;">
                    {title}
                </div>
            </div>
        </div>
        """
        
        # Define footer HTML
        footer_html = """
        <div style="background:#003366; color:white; padding:10px; text-align:center;
                    font-size:12px; font-family:'BC Sans', sans-serif; position:fixed;
                    bottom:0; width:100%; z-index:9999;">
            © 2025 Government of British Columbia | INTERNAL GOVERNMENT USE ONLY
        </div>
        </body>"""
        
        # Inject into HTML
        html = html.replace('<body>', f'<body>\n{header_html}')
        html = html.replace('</body>', footer_html)
        
        # Write modified HTML
        with map_path.open('w', encoding='utf-8') as f:
            f.write(html)
    
    def _get_color_for_values(
        self,
        unique_values: List[str],
        use_random: bool = False
    ) -> Dict[str, str]:
        """
        Generate color mapping for unique values.
        
        Args:
            unique_values: List of unique values to color
            use_random: Whether to use random colors (vs palette)
        
        Returns:
            Dictionary mapping values to colors
        """
        color_mapping = {}
        
        if use_random or len(unique_values) > len(self.COLOR_PALETTE):
            # Use random colors
            for value in unique_values:
                color = f"#{np.random.randint(0, 255):02X}{np.random.randint(0, 255):02X}{np.random.randint(0, 255):02X}"
                color_mapping[value] = color
        else:
            # Use palette
            for i, value in enumerate(unique_values):
                color_mapping[value] = self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]
        
        return color_mapping
    
    def initialize_all_layers_map(self) -> None:
        """Initialize the all-layers combined map."""
        print('....Initializing all-layers map')
        self.map_all = self._create_map_template()
        self.aoi_groups_all = self._add_aoi_layers(self.map_all, show_buffers=True)
        self.category_groups = {'AREA OF INTEREST': self.aoi_groups_all}
    
    def create_individual_map(
        self,
        gdf_intersect: gpd.GeoDataFrame,
        label_col: str,
        item_name: str,
        category: str,
        data_source: str = ""
    ) -> None:
        """
        Create an individual HTML map for a dataset with overlaps.
        
        Args:
            gdf_intersect: GeoDataFrame of intersecting features
            label_col: Column name for labeling features
            item_name: Name of the dataset/layer
            category: Category name for grouping
            data_source: Data source description
        """
        if gdf_intersect.empty:
            print(f'.......No features to map for {item_name}')
            return
        
        # Prepare GeoDataFrame
        gdf = gdf_intersect.copy()
        
        # Ensure WGS84 for mapping
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Flatten 3D geometries if present
        if gdf.geometry.has_z.any():
            from shapely import wkt
            gdf['geometry'] = gdf['geometry'].apply(
                lambda geom: wkt.loads(wkt.dumps(geom, output_dimension=2))
            )
        
        # Convert columns to string for display
        for col in gdf.columns:
            if col != 'geometry':
                gdf[col] = gdf[col].astype(str)
        
        # Add metadata columns
        map_title = item_name.replace('_', ' ')
        gdf['layer_name'] = map_title
        gdf['category'] = str(category) if category and not pd.isna(category) else 'Uncategorized'
        gdf['data_source'] = str(data_source) if data_source and not pd.isna(data_source) else 'N/A'
        
        # Determine color mapping based on unique values
        if label_col in gdf.columns:
            unique_values = gdf[label_col].unique()
        else:
            unique_values = [map_title]
            label_col = 'layer_name'
        
        # Limit legend entries
        max_legend_items = 20
        if len(unique_values) > max_legend_items:
            # Use single color for all
            gdf['__color__'] = '#1f77b4'
            legend_items = [('#1f77b4', map_title)]
            legend_header = "Legend"
        else:
            # Assign colors to unique values
            color_mapping = self._get_color_for_values(list(unique_values))
            gdf['__color__'] = gdf[label_col].map(color_mapping)
            legend_items = [(color, str(val)) for val, color in color_mapping.items()]
            legend_header = label_col
        
        # Get bounds for this layer
        xmin, ymin, xmax, ymax = gdf.geometry.total_bounds
        layer_bounds = (ymin, xmin, ymax, xmax)
        
        # Create individual map
        map_one = self._create_map_template(fit_bounds=layer_bounds)
        
        # Add AOI and buffers
        aoi_groups = self._add_aoi_layers(map_one, show_buffers=True)
        
        # Create feature groups for each unique value (for grouped layer control)
        layer_groups = []
        tooltip_fields = ['layer_name', 'data_source']
        
        for value in unique_values:
            gdf_subset = gdf[gdf[label_col] == value]
            color = gdf_subset['__color__'].iloc[0] if not gdf_subset.empty else '#1f77b4'
            
            grp = folium.FeatureGroup(name=str(value), show=True)
            
            folium.GeoJson(
                data=gdf_subset,
                name=str(value),
                style_function=lambda x, c=color: {
                    'fillColor': c,
                    'color': c,
                    'weight': 2,
                    'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=['Layer', 'Source'],
                    labels=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=[col for col in gdf_subset.columns 
                            if col not in ['geometry', '__color__', 'layer_name', 'category', 'data_source']],
                    labels=True
                )
            ).add_to(grp)
            
            grp.add_to(map_one)
            layer_groups.append(grp)
        
        # Add legend
        legend_html = self._generate_legend_html(legend_items, legend_header, include_aoi=True)
        map_one.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl(collapsed=True).add_to(map_one)
        
        # Add grouped layer control for AOI
        GroupedLayerControl(
            groups={"AREA OF INTEREST": aoi_groups},
            exclusive_groups=False,
            collapsed=True
        ).add_to(map_one)
        
        # Add grouped layer control for feature layers
        if layer_groups:
            GroupedLayerControl(
                groups={map_title.upper(): layer_groups},
                exclusive_groups=False,
                collapsed=True
            ).add_to(map_one)
        
        # Save map
        out_path = os.path.join(self.maps_dir, f'{item_name}.html')
        map_one.save(out_path)
        
        # Inject branding
        self._inject_branding(out_path, title=map_title)
        
        # Also add to all-layers map
        self._add_to_all_layers_map(gdf, map_title, category, tooltip_fields)
    
    def _add_to_all_layers_map(
        self,
        gdf: gpd.GeoDataFrame,
        layer_name: str,
        category: str,
        tooltip_fields: List[str]
    ) -> None:
        """
        Add a layer to the all-layers combined map.
        
        Args:
            gdf: GeoDataFrame to add
            layer_name: Name for the layer
            category: Category for grouping
            tooltip_fields: Fields to show in tooltip
        """
        if self.map_all is None:
            self.initialize_all_layers_map()
        
        # Assign a unique color from the palette using the color index
        color = self.COLOR_PALETTE[self._color_index % len(self.COLOR_PALETTE)]
        self._color_index += 1
        
        # Track for legend
        self.all_layers_legend.append((color, layer_name))
        
        grp = folium.FeatureGroup(name=layer_name, show=False)
        
        # Get actual columns in the GeoDataFrame (excluding geometry)
        available_cols = [col for col in gdf.columns if col != 'geometry']
        
        # Filter tooltip fields to only include columns that exist
        valid_tooltip_fields = [f for f in tooltip_fields if f in available_cols]
        
        # Determine popup fields (exclude internal columns)
        popup_fields = [
            col for col in available_cols 
            if col not in ['__color__', 'layer_name', 'category', 'data_source']
        ]
        
        # Limit popup fields and ensure they exist
        popup_fields = popup_fields[:10] if popup_fields else valid_tooltip_fields
        
        # Build the GeoJson layer
        geojson_layer = folium.GeoJson(
            data=gdf,
            name=layer_name,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 2,
                'fillOpacity': 0.4
            }
        )
        
        # Add tooltip if we have valid fields
        if valid_tooltip_fields:
            geojson_layer.add_child(
                folium.GeoJsonTooltip(
                    fields=valid_tooltip_fields,
                    aliases=['Layer', 'Source'][:len(valid_tooltip_fields)],
                    labels=True
                )
            )
        
        # Add popup if we have valid fields
        if popup_fields:
            geojson_layer.add_child(
                folium.GeoJsonPopup(
                    fields=popup_fields,
                    labels=True
                )
            )
        
        geojson_layer.add_to(grp)
        grp.add_to(self.map_all)
        
        # Track by category - ensure category is a string and handle NaN/None
        if category and isinstance(category, str) and category.strip():
            category_upper = category.upper()
        else:
            category_upper = 'UNCATEGORIZED'
        if category_upper not in self.category_groups:
            self.category_groups[category_upper] = []
        self.category_groups[category_upper].append(grp)
    
    def save_all_layers_map(self) -> None:
        """Save the all-layers combined map."""
        if self.map_all is None:
            print('....No layers added to all-layers map')
            return
        
        print('....Saving all-layers map')
        
        # Build legend data as JavaScript array
        legend_data_js = "[\n"
        for color, layer_name in self.all_layers_legend:
            # Escape quotes in layer names
            escaped_name = layer_name.replace("'", "\\'").replace('"', '\\"')
            legend_data_js += f'        {{color: "{color}", name: "{escaped_name}"}},\n'
        legend_data_js += "    ]"
        
        # Create collapsible legend with dynamic visibility based on layer toggles
        legend_html = f'''
        <div id="legend-container" style="position: fixed; 
            bottom: 60px; right: 10px; z-index: 1000;">
            
            <!-- Toggle Button -->
            <button id="legend-toggle" onclick="toggleLegend()" style="
                background-color: #003366; color: white; border: none;
                padding: 8px 12px; border-radius: 4px; cursor: pointer;
                font-family: 'BC Sans', Arial, sans-serif; font-size: 12px;
                margin-bottom: 5px; display: block; margin-left: auto;">
                ▼ Legend
            </button>
            
            <!-- Legend Panel -->
            <div id="legend-panel" style="
                background-color: #fff; padding: 10px; 
                border-radius: 5px; border: 1px solid grey;
                font-family: 'BC Sans', Arial, sans-serif;
                max-height: 300px; overflow-y: auto;
                max-width: 250px; display: block;">
                
                <div style="font-weight: bold; margin-bottom: 8px; 
                            border-bottom: 1px solid #ccc; padding-bottom: 5px;">
                    Active Layers
                </div>
                
                <!-- Static AOI entries -->
                <div style="margin: 4px 0;">
                    <span style="display: inline-block; 
                        margin-right: 8px; background-color: transparent;
                        border: 2px solid red;
                        width: 15px; height: 15px; vertical-align: middle;"></span>
                    <span style="vertical-align: middle;">AOI</span>
                </div>
                <div style="margin: 4px 0;">
                    <span style="display: inline-block; 
                        margin-right: 8px; background-color: transparent;
                        border: 2px dashed orange;
                        width: 15px; height: 15px; vertical-align: middle;"></span>
                    <span style="vertical-align: middle;">AOI Buffers</span>
                </div>
                
                <!-- Dynamic layer entries -->
                <div id="dynamic-legend" style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 8px;">
                    <div style="font-size: 11px; color: #666;">Toggle layers to see them here</div>
                </div>
            </div>
        </div>
        
        <script>
        var legendData = {legend_data_js};
        var legendCollapsed = false;
        
        function toggleLegend() {{
            var panel = document.getElementById('legend-panel');
            var btn = document.getElementById('legend-toggle');
            legendCollapsed = !legendCollapsed;
            if (legendCollapsed) {{
                panel.style.display = 'none';
                btn.innerHTML = '► Legend';
            }} else {{
                panel.style.display = 'block';
                btn.innerHTML = '▼ Legend';
            }}
        }}
        
        function updateDynamicLegend() {{
            var container = document.getElementById('dynamic-legend');
            var html = '';
            var activeLayers = 0;
            
            // Check which layers are active by looking at checkboxes in layer control
            var inputs = document.querySelectorAll('.leaflet-control-layers-overlays input');
            
            inputs.forEach(function(input, index) {{
                if (input.checked) {{
                    // Find corresponding legend data
                    var layerName = input.parentElement.textContent.trim();
                    var layerInfo = legendData.find(function(item) {{
                        return item.name === layerName;
                    }});
                    
                    if (layerInfo) {{
                        var displayName = layerInfo.name.length > 30 ? 
                            layerInfo.name.substring(0, 30) + '...' : layerInfo.name;
                        html += '<div style="margin: 3px 0; font-size: 11px;">' +
                            '<span style="display: inline-block; margin-right: 6px; ' +
                            'background-color: ' + layerInfo.color + '; ' +
                            'width: 12px; height: 12px; vertical-align: middle;"></span>' +
                            '<span style="vertical-align: middle;" title="' + layerInfo.name + '">' + 
                            displayName + '</span></div>';
                        activeLayers++;
                    }}
                }}
            }});
            
            if (activeLayers === 0) {{
                html = '<div style="font-size: 11px; color: #666;">Toggle layers to see them here</div>';
            }}
            
            container.innerHTML = html;
        }}
        
        // Update legend when layer visibility changes
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                var inputs = document.querySelectorAll('.leaflet-control-layers input');
                inputs.forEach(function(input) {{
                    input.addEventListener('change', updateDynamicLegend);
                }});
                updateDynamicLegend();
            }}, 500);
        }});
        </script>
        '''
        
        self.map_all.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control (collapsed to avoid overlap)
        folium.LayerControl(collapsed=True).add_to(self.map_all)
        
        # Add grouped layer control by category
        if self.category_groups:
            GroupedLayerControl(
                groups=self.category_groups,
                exclusive_groups=False,
                collapsed=True
            ).add_to(self.map_all)
        
        # Save map
        out_path = os.path.join(self.maps_dir, '00_all_layers.html')
        self.map_all.save(out_path)
        
        # Inject branding
        self._inject_branding(out_path, title='Overview Map - All Overlaps')

# ============================================================================
# CONVENIENCE FUNCTION FOR SIMPLE USE
# ============================================================================

def create_status_map(
    gdf_aoi: gpd.GeoDataFrame,
    gdf_intersect: gpd.GeoDataFrame,
    label_col: str,
    item_name: str,
    workspace: str,
    category: str = "",
    data_source: str = ""
) -> None:
    """
    Simple function to create a single status map (backward compatible).
    
    Args:
        gdf_aoi: Area of Interest GeoDataFrame
        gdf_intersect: Intersecting features GeoDataFrame
        label_col: Column for labeling
        item_name: Dataset name
        workspace: Output workspace
        category: Optional category name
        data_source: Optional data source description
    """
    # Create a minimal df_stat for buffer creation
    df_stat = pd.DataFrame({'Buffer_Distance': [0]})
    
    generator = MapGenerator(gdf_aoi, workspace, df_stat)
    generator.create_individual_map(
        gdf_intersect, 
        label_col, 
        item_name, 
        category,
        data_source
    )