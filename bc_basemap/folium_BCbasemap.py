import folium
from folium import MacroElement
from jinja2 import Template

# Define a custom MacroElement to inject your JS code
class ESRIVectorTile(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = Template(u"""
            {% macro script(this, kwargs) %}
            // Configure Mapbox GL layer with error handling
            var glLayer = L.mapboxGL({
                style: 'https://www.arcgis.com/sharing/rest/content/items/b1624fea73bd46c681fab55be53d96ae/resources/styles/root.json',
                accessToken: 'no-token-needed'
            }).addTo({{ this._parent.get_name() }});
            
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
                
                // Check for label layers
                var labelLayers = mapGL.getStyle().layers.filter(function(l) {
                    return l.type === 'symbol' && l.layout && l.layout['text-field'];
                });
                
                console.log('Label layers found:', labelLayers);
                if (labelLayers.length === 0) {
                    console.warn('No label layers found in style');
                }
            }
            
            // Optional: Add layer inspection control
            L.Control.LayerInspector = L.Control.extend({
                onAdd: function(map) {
                    var div = L.DomUtil.create('div', 'leaflet-control-layer-inspector');
                    div.innerHTML = '<button>Show Layers</button>';
                    div.firstChild.onclick = function() { debugLabelLayers(); };
                    return div;
                }
            });
            new L.Control.LayerInspector({ position: 'topright' }).addTo({{ this._parent.get_name() }});
            {% endmacro %}
        """)

# Create the Folium map with no default tiles (we'll add our vector tile layer)
m = folium.Map(location=[49.2827, -123.1207], zoom_start=12, tiles=None)

# (Optional) Add custom CSS to ensure the map container fills the page
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

# Add the Mapbox GL JS and leaflet-mapbox-gl dependencies
js_css = [
    '<link href="https://api.mapbox.com/mapbox-gl-js/v2.9.2/mapbox-gl.css" rel="stylesheet">',
    '<script src="https://api.mapbox.com/mapbox-gl-js/v2.9.2/mapbox-gl.js"></script>',
    '<script src="https://unpkg.com/mapbox-gl-leaflet/leaflet-mapbox-gl.js"></script>'
]
for item in js_css:
    m.get_root().html.add_child(folium.Element(item))

# Inject our custom JS code (the ESRI vector tile layer and error/debug functions)
m.add_child(ESRIVectorTile())

# Add attribution as a fixed overlay in the bottom right corner
attribution_html = """
<div style="position: absolute; bottom: 10px; right: 10px; z-index: 9999;
            background: rgba(255, 255, 255, 0.8); padding: 5px; font-size: 10px;">
Tiles © <a target='_blank' href='https://catalogue.data.gov.bc.ca/dataset/78895ec6-c679-4837-a01a-8d65876a3da9'>
ESRI &amp; GeoBC</a>
</div>
"""
m.get_root().html.add_child(folium.Element(attribution_html))

# Save the map to an HTML file
m.save("BCbasemap_vector_tile.html")
