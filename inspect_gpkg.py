import fiona
import sys

gpkg_path = sys.argv[1]

try:
    layers = fiona.listlayers(gpkg_path)
    print(f"Layers in {gpkg_path}: {layers}")

    for layer_name in layers:
        with fiona.open(gpkg_path, layer=layer_name) as src:
            print(f"\nLayer: {layer_name}")
            print(f"Schema properties (fields): {sorted(src.schema.get('properties', {}).keys())}")
except Exception as e:
    print(f"Error inspecting {gpkg_path}: {e}", file=sys.stderr)
