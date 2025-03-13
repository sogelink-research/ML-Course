import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

warnings.simplefilter(
    "ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning
)


def download_ahn():
    print("Downloading AHN5 DSM 50cm...")
    url = "https://ns_hwh.fundaments.nl/hwh-ahn/AHN5/03a_DSM_50cm/2023_R_25GN1.TIF"
    output_file = "data/25GN1.tif"

    # Download the file if it doesn't already exist
    if not os.path.exists(output_file):
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Done.")
    return output_file


# def download_luchtfoto(bbox_str: str):
#     print("Downloading luchtfoto 25cm...")
#     output_file = "data/25GN1_luchtfoto.tif"

#     url = "https://ns_hwh.fundaments.nl/hwh-ortho/2023/Ortho/1/04/beelden_tif_tegels/2023_153000_464000_RGB_hrl.tif"
#     # Download the file if it doesn't already exist
#     if not os.path.exists(output_file):
#         response = requests.get(url, stream=True, verify=False)
#         response.raise_for_status()
#         with open(output_file, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)

#     print("Done.")
#     return output_file


# def download_luchtfoto(tif_file: Path, bbox_str: str):
#     print("Downloading luchtfoto 25cm...")
#     tif_file.parent.mkdir(parents=True, exist_ok=True)
#     if not tif_file.exists():
#         wmts_url = f"https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?&request=GetCapabilities&service=WMTS&bbox={bbox_str}"
#         layer_name = "Actueel_ortho25"

#         # Fetch data from WFS and load into GeoDataFrame
#         gdf = gpd.read_file(f"{wmts_url}&typeName={layer_name}")

#         # Write the GeoDataFrame to a TIF file
#         gdf.to_file(tif_file, driver="GTiff")
#     print("Done.")


def clip_image(input_file: str, output_file: str, bbox_shapely: box):
    print(f"Clipping {input_file} to bbox {bbox_shapely.bounds}...")
    with rasterio.open(input_file) as src:
        # Clip the image to the bbox
        out_image, out_transform = mask(src, [bbox_shapely], crop=True)

        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        # Write the clipped image to a new file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(out_image)
    print("Done.")


def download_buildings(geojson_file: Path, bbox_str: str):
    print("Downloading BAG buildings...")
    geojson_file.parent.mkdir(parents=True, exist_ok=True)
    if not geojson_file.exists():
        wfs_url = f"WFS:https://service.pdok.nl/lv/bag/wfs/v2_0?request=getCapabilities&service=WFS&bbox={bbox_str}"
        layer_name = "bag:pand"

        # Fetch data from WFS and load into GeoDataFrame
        gdf = gpd.read_file(f"{wfs_url}&typeName={layer_name}")

        # Write the GeoDataFrame to a GeoJSON file
        gdf.to_file(geojson_file, driver="GeoJSON")
    print("Done.")


def create_mask(geojson_file: str, mask_file: str, bbox_shapely: box):
    print(f"Creating mask from {geojson_file}...")
    # Load the vector data from GeoJSON file
    gdf = gpd.read_file(geojson_file)

    # Define the raster properties
    pixel_size = 0.5

    # Get the bounds of the vector data
    minx, miny, maxx, maxy = gdf.total_bounds

    # Define the transformation (top-left corner, pixel size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # Calculate the dimensions of the output raster
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    # Create an empty raster with the given dimensions and transform
    temp_file = "temp.tiff"
    with rasterio.open(
        temp_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,  # One band
        dtype="uint8",  # Unsigned 8-bit integer
        crs=gdf.crs,  # Coordinate reference system
        transform=transform,
        nodata=255,
    ) as dst:
        # Rasterize the geometries into the raster band
        mask = geometry_mask(
            gdf.geometry, transform=transform, invert=True, out_shape=(height, width)
        )
        band = np.zeros((height, width), dtype=np.float32)
        band[mask] = 1

        # Write the modified band back into the raster
        dst.write(band, 1)

    # Clip the mask to the bbox
    clip_image(temp_file, mask_file, bbox_shapely)

    # Remove the temporary file
    os.remove(temp_file)

    print(f"Rasterized mask saved to {mask_file}")


def tile_tiff_rasterio(image_path, output_folder, tile_size=512):
    print(f"Creating tiles from {image_path}...")
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the TIFF file
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height

        tile_index = 0
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                # Define the window for the current tile
                window = Window(
                    j, i, min(tile_size, width - j), min(tile_size, height - i)
                )

                # Read the windowed tile
                tile = src.read(window=window)

                # Define metadata for the new tile
                tile_meta = src.meta.copy()
                tile_meta.update(
                    {
                        "width": window.width,
                        "height": window.height,
                        "transform": rasterio.windows.transform(window, src.transform),
                    }
                )

                # Save tile
                tile_path = os.path.join(output_folder, f"tile_{tile_index}.tif")
                with rasterio.open(tile_path, "w", **tile_meta) as dest:
                    dest.write(tile)

                tile_index += 1

    print(f"Saved {tile_index} tiles in {output_folder}")


def merge_tiff_files(input_files: list[Path], output_file: Path):
    src_files_to_mosaic = [rasterio.open(f, "r") for f in input_files]
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=src_files_to_mosaic[0].nodata)

    # Define output metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )

    # Write the merged raster to a new file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close the source files
    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster saved as {output_file}")


if __name__ == "__main__":

    minx, miny, maxx, maxy = 120000.0, 487500.0, 125000.0, 481250.0
    bbox = (minx, miny, maxx, maxy)
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    bbox_shapely = box(minx, miny, maxx, maxy)

    # os.makedirs("data", exist_ok=True)

    # download_buildings(Path("test_buildings.geojson"), bbox_str)
    # download_luchtfoto(Path("test_luchtfoto.tif"), bbox_str)

    # download_luchtfoto(bbox_str)

    # full_dsm_file = download_ahn()
    # full_image_file = download_luchtfoto()
    # cropped_dsm_file = "data/dsm/merged.tif"
    # clip_image(full_dsm_file, cropped_dsm_file, bbox_shapely)
    # geojson_file = "data/buildings/panden.geojson"
    # download_buildings(geojson_file)
    # cropped_mask_file = "data/mask/merged.tif"
    # create_mask(geojson_file, cropped_mask_file, bbox_shapely)
    # tile_tiff_rasterio(cropped_dsm_file, "data/image/tiles")
    # tile_tiff_rasterio(cropped_mask_file, "data/mask/tiles")
    # os.remove(full_dsm_file)

    from io import BytesIO
    from pathlib import Path

    import requests
    from owslib.wmts import WebMapTileService
    from PIL import Image

    def download_wmts_image(image_file: Path, bbox: tuple, zoom_level: int = 12):
        print("Downloading aerial image from WMTS...")

        # Define the WMTS URL
        wmts_url = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?"

        # Connect to the WMTS service
        wmts = WebMapTileService(wmts_url)

        # Define layer name and tile matrix set (Check capabilities for correct values)
        layer_name = "Actueel_ortho25"
        tile_matrix_set = "EPSG:3857"  # Ensure this matches the expected CRS

        # Convert bbox to tile coordinates (this is a simplified example)
        min_x, min_y, max_x, max_y = bbox
        tile_col = int((min_x + 20037508.34) / (40075016.68 / (2**zoom_level)))
        tile_row = int((20037508.34 - max_y) / (40075016.68 / (2**zoom_level)))

        # Construct tile request URL
        tile_url = (
            f"{wmts_url}service=WMTS&request=GetTile&version=1.0.0"
            f"&layer={layer_name}&style=&tilematrixset={tile_matrix_set}"
            f"&tilematrix={zoom_level}&tilerow={tile_row}&tilecol={tile_col}"
            f"&format=image/jpeg"
        )

        # Download the image
        response = requests.get(tile_url)
        if response.status_code == 200:
            image_file.parent.mkdir(parents=True, exist_ok=True)
            image = Image.open(BytesIO(response.content))
            image.save(image_file, "JPEG")
            print("Image downloaded and saved successfully.")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")

    # Example usage
    image_path = Path("output/aerial_image.jpg")
    # bbox = (
    #     5.121420,
    #     52.090736,
    #     5.122420,
    #     52.091736,
    # )  # Example bounding box (longitude, latitude)
    download_wmts_image(image_path, bbox, zoom_level=14)
