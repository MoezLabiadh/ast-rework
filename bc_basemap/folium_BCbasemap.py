import folium
from folium import MacroElement
from jinja2 import Template

# Define a custom MacroElement to inject JavaScript code
class ESRIVectorTile(MacroElement):
    def __init__(self):
        super().__init__()
        # The template uses the parent FeatureGroup's name as the layer container.
        self._template = Template(u"""
            {% macro script(this, kwargs) %}
            // Create the Mapbox GL layer for the ESRI vector tile
            var glLayer = L.mapboxGL({
                style: 'https://www.arcgis.com/sharing/rest/content/items/b1624fea73bd46c681fab55be53d96ae/resources/styles/root.json',
                accessToken: 'no-token-needed'
            });
            // Add the glLayer to its parent (the FeatureGroup)
            glLayer.addTo({{ this._parent.get_name() }});
            
            // Add error listeners
            glLayer.getGLMap().on('load', function() {
                console.log('Mapbox GL map loaded');
                debugLabelLayers();
            });
            glLayer.getGLMap().on('error', function(e) {
                console.error('Mapbox GL error:', e.error);
            });
            
            // Debug function to check label layers
            function debugLabelLayers() {
                var mapGL = glLayer.getGLMap();
                console.log('All layers:', mapGL.getStyle().layers);
                var labelLayers = mapGL.getStyle().layers.filter(function(l) {
                    return l.type === 'symbol' && l.layout && l.layout['text-field'];
                });
                console.log('Label layers found:', labelLayers);
                if (labelLayers.length === 0) {
                    console.warn('No label layers found in style');
                }
            }
            
            // Optional: Add a layer inspection control to the main map
            L.Control.LayerInspector = L.Control.extend({
                onAdd: function(map) {
                    var div = L.DomUtil.create('div', 'leaflet-control-layer-inspector');
                    div.innerHTML = '<button>Show Layers</button>';
                    div.firstChild.onclick = function() { debugLabelLayers(); };
                    return div;
                }
            });
            new L.Control.LayerInspector({ position: 'topright' }).addTo({{ this._parent.get_root().get_name() }});
            {% endmacro %}
        """)

# Create the Folium map without a default tile layer (we'll use our custom vector tile)
m = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles=None)

# Add custom CSS to ensure the map fills the page and to set z-index for the Mapbox GL canvas
css = """
<style>
  html, body, #map {
      height: 100%;
      margin: 0;
      padding: 0;
  }
  .leaflet-mapbox-gl-layer {
      z-index: 1;
  }
</style>
"""
m.get_root().header.add_child(folium.Element(css))

# Include external JS/CSS dependencies for Mapbox GL and the leaflet-mapbox-gl plugin
dependencies = [
    '<link href="https://api.mapbox.com/mapbox-gl-js/v2.9.2/mapbox-gl.css" rel="stylesheet">',
    '<script src="https://api.mapbox.com/mapbox-gl-js/v2.9.2/mapbox-gl.js"></script>',
    '<script src="https://unpkg.com/mapbox-gl-leaflet/leaflet-mapbox-gl.js"></script>'
]
for dep in dependencies:
    m.get_root().html.add_child(folium.Element(dep))

# Create a FeatureGroup for the ESRI vector tile layer so it appears in the layer control
vector_tile_fg = folium.FeatureGroup(name="ESRI Vector Tile")
vector_tile_fg.add_child(ESRIVectorTile())
m.add_child(vector_tile_fg)

# Add a LayerControl to toggle overlays (and any base layers if added)
folium.LayerControl(collapsed=False).add_to(m)

# Save the final map to an HTML file
m.save("BCbasemap_vector_tile.html")
m