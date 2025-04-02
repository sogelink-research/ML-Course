import json
import warnings
from math import ceil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from owslib.wfs import WebFeatureService
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import Window

# from shapely.geometry import box
from tqdm.auto import tqdm

from simple_model.bbox import BboxInt

warnings.simplefilter(
    "ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning
)


def find_dsm_urls_for_bbox(main_data_folder: Path, bbox: BboxInt):
    """
    Finds the URLs corresponding to the parts of the AHN5 DSM that intersect with the given bounding box.

    Parameters
    ----------
    main_data_folder : Path
        The folder where the data is stored.
    bbox : BboxInt
        The bounding box.

    Returns
    -------
    list[str]
        The list of URLs to the AHN5 DSM 50cm files.
    """
    # Download the GeoPackage with the regions
    regions_url = "https://ns_hwh.fundaments.nl/hwh-ahn/AUX/bladwijzer.gpkg"
    regions_file = main_data_folder / "bladwijzer.gpkg"

    if not regions_file.exists():
        regions_file.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(regions_url, stream=True, verify=False)
        response.raise_for_status()
        with open(regions_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # Load the GeoPackage into a GeoDataFrame
    regions_gdf = gpd.read_file(regions_file)

    # Find the regions that intersect with the bbox
    retraction_value = 0.25
    bbox_shapely = (
        bbox.to_float().buffer(-retraction_value, -retraction_value).to_shapely()
    )

    intersecting_regions = regions_gdf[regions_gdf.intersects(bbox_shapely)]

    # Find the AHN urls for the intersecting regions
    ahn_urls = []
    for index, row in intersecting_regions.iterrows():
        ahn_urls.append(row["AHN5_05M_R"])

    print(f"Found {len(ahn_urls)} AHN5 DSM 50cm URLs for the bounding box.")

    return ahn_urls


def download_dsm(tif_file: Path, bbox: BboxInt, main_data_folder: Path):
    """
    Download the AHN5 DSM 50cm for the given bounding box.

    Parameters
    ----------
    tif_file : Path
        The path to the output TIF file.
    bbox : BboxInt
        The bounding box.
    main_data_folder : Path
        The folder where the data is stored.

    Returns
    -------
    None
    """
    print("Downloading AHN5 DSM 50cm...")
    if tif_file.exists():
        print("File already exists.")
        return

    urls = find_dsm_urls_for_bbox(main_data_folder, bbox)

    # Download the files
    temp_files = []
    tif_file.parent.mkdir(parents=True, exist_ok=True)

    for i, url in tqdm(enumerate(urls), leave=False):
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()

        temp_file = tif_file.parent / f"temp_{i}.tif"
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Clip the image to the bbox
        clip_image(temp_file, temp_file, bbox)

        temp_files.append(temp_file)

    # Merge the temporary files into a single file
    merge_tiff_files(temp_files, tif_file, nodata=-9999)

    # Remove the temporary files
    for temp_file in temp_files:
        temp_file.unlink()


# def download_dsm_old_1(data_folder: Path) -> Path:
#     print("Downloading AHN5 DSM 50cm...")
#     url = "https://ns_hwh.fundaments.nl/hwh-ahn/AHN5/03a_DSM_50cm/2023_R_25GN1.TIF"
#     output_file = data_folder / "25GN1.tif"

#     # Download the file if it doesn't already exist
#     if not output_file.exists():
#         response = requests.get(url, stream=True, verify=False)
#         response.raise_for_status()
#         with open(output_file, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)

#     print("Done.")
#     return output_file

# def download_dsm_old_2(tif_file: Path, bbox: tuple[int, int, int, int]):
#     print("Downloading DSM 50cm...")
#     # Get the WCS of the AHN
#     # url = "https://service.pdok.nl/rws/ahn/wcs/v1_0?request=GetCapabilities&service=WCS"
#     url = "https://api.ellipsis-drive.com/v3/ogc/wcs/a4a8a27b-e36e-4dd5-a75b-f7b6c18d33ec?request=getCapabilities&version=1.0.0"
#     wcs = WebCoverageService(url=url, version="1.0.0")

#     print(list(wcs.contents))
#     # layer = "dsm_05m"
#     layer = "fc9d369f-94ca-4373-8281-a6854edb67c9"

#     # Take the 0.5m DSM as an example
#     cvg = wcs.contents[layer]

#     # Print supported reference systems, the bounding box defined in WGS 84 coordinates, and supported file formats
#     print(cvg.supportedCRS)
#     print(cvg.boundingBoxWGS84)
#     print(cvg.supportedFormats)

#     step_size = 500
#     temp_tifs = []
#     current_index = 0

#     for xmin in range(floor(bbox[0]), ceil(bbox[2]), step_size):
#         for ymin in range(floor(bbox[3]), ceil(bbox[1]), step_size):
#             xmax = min(xmin + step_size, int(bbox[2]))
#             ymax = min(ymin + step_size, int(bbox[1]))
#             bbox_correct = (xmin, ymin, xmax, ymax)

#             # Get the coverage for the study area
#             # response = wcs.getCoverage(
#             #     identifier=layer,
#             #     bbox=bbox_correct,
#             #     format="GEOTIFF",
#             #     crs="urn:ogc:def:crs:EPSG::28992",
#             #     resx=0.5,
#             #     resy=0.5,
#             # )

#             width = 2 * (bbox_correct[2] - bbox_correct[0])
#             height = 2 * (bbox_correct[3] - bbox_correct[1])
#             print(width, height)

#             # Get the coverage for the study area
#             response = wcs.getCoverage(
#                 identifier=layer,
#                 bbox=bbox_correct,
#                 format="GEOTIFF",
#                 crs="urn:ogc:def:crs:EPSG::28992",
#                 # resx=0.5,
#                 # resy=0.5,
#                 width=width,
#                 height=height,
#             )

#             # Save the response to a file
#             current_tif = tif_file.parent / f"temp_{current_index}.tif"
#             with open(current_tif, "wb") as f:
#                 f.write(response.read())

#             temp_tifs.append(current_tif)
#             current_index += 1

#     # Merge the temporary files into a single file
#     merge_tiff_files(temp_tifs, tif_file, nodata=-9999)
#     for temp_tif in temp_tifs:
#         temp_tif.unlink()


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


def clip_image(input_file: Path, output_file: Path, bbox: BboxInt):
    bbox_shapely = bbox.to_shapely()
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
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(out_image)
    print("Done.")


# def download_buildings_old(geojson_file: Path, bbox_str: str):
#     print("Downloading BAG buildings...")
#     geojson_file.parent.mkdir(parents=True, exist_ok=True)
#     if not geojson_file.exists() or True:
#         wfs_url = f"WFS:https://service.pdok.nl/lv/bag/wfs/v2_0?request=getCapabilities&service=WFS&bbox={bbox_str}"
#         layer_name = "bag:pand"

#         # Fetch data from WFS and load into GeoDataFrame
#         gdf = gpd.read_file(f"{wfs_url}&typeName={layer_name}")

#         # Write the GeoDataFrame to a GeoJSON file
#         gdf.to_file(geojson_file, driver="GeoJSON")
#     print("Done.")


def split_bbox(bbox: BboxInt, max_size: int) -> list[BboxInt]:
    """
    Splits the bounding box into smaller parts of the given size.

    Parameters
    ----------
    bbox : BboxInt
        The bounding box.
    max_size : int
        The maximum size of the parts.

    Returns
    -------
    list[BboxInt]
        The list of smaller bounding boxes.
    """
    minx, maxy, maxx, miny = bbox.minx, bbox.maxy, bbox.maxx, bbox.miny
    width = maxx - minx
    height = maxy - miny

    nx = ceil(width / max_size)
    ny = ceil(height / max_size)

    bboxes = []

    for i in range(nx):
        for j in range(ny):
            new_minx = minx + i * max_size
            new_miny = miny + j * max_size
            new_maxx = min(minx + (i + 1) * max_size, maxx)
            new_maxy = min(miny + (j + 1) * max_size, maxy)

            new_bbox = BboxInt(new_minx, new_miny, new_maxx, new_maxy, bbox.y_up)
            bboxes.append(new_bbox)

    return bboxes


def download_buildings(geojson_file: Path, bbox: BboxInt):
    print("Downloading BAG buildings...")

    if geojson_file.exists():
        print("File already exists.")
        return

    # Get the WFS of the BAG
    wfsUrl = "https://service.pdok.nl/lv/bag/wfs/v2_0"
    wfs = WebFeatureService(url=wfsUrl, version="2.0.0")
    layer = list(wfs.contents)[0]

    # Parameters for splitting the bounding box
    max_size = 2000  # Maximum size of the bounding box
    bboxes = split_bbox(bbox, max_size)

    # Number of features per request
    count = 1000

    all_features = []

    for sub_bbox in tqdm(bboxes, leave=False):
        start_index = 0
        while True:
            response = wfs.getfeature(
                typename=layer,
                bbox=sub_bbox.to_tuple(),
                outputFormat="json",
                maxfeatures=count,
                startindex=start_index,
            )
            data = json.loads(response.read())

            features = data.get("features", [])
            if not features:
                break  # Stop when there are no more features to fetch

            all_features.extend(features)
            start_index += count  # Move to the next batch

    # Create GeoDataFrame, without saving first
    buildings_gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:28992")

    # Save the GeoDataFrame to a GeoJSON file
    geojson_file.parent.mkdir(parents=True, exist_ok=True)
    buildings_gdf.to_file(geojson_file, driver="GeoJSON")

    print(f"Downloaded BAG buildings to {geojson_file}")


# def download_buildings(geojson_file: Path, bbox_str: str):
#     url = "https://service.pdok.nl/lv/bag/wfs/v2_0"

#     # Specify parameters (read data in json format).
#     params = dict(
#         service="WFS",
#         version="2.0.0",
#         request="GetFeature",
#         typeName="bag:pand",
#         outputFormat="json",
#         bbox=bbox_str,
#     )

#     r = requests.get(url, params=params)

#     data = gpd.GeoDataFrame.from_features(geojson.loads(r.content), crs="EPSG:28992")
#     data.to_file(geojson_file, driver="GeoJSON")
#     print(f"Downloaded BAG buildings to {geojson_file}")


def filter_small_buildings(input_file: Path, output_file: Path):
    """
    Filter out the buildings that have the property "gebruiksdoel" set to NULL
    and an area smaller than 30 m².
    """
    print("Filtering buildings...")

    # Load the GeoDataFrame
    gdf = gpd.read_file(input_file)

    # Filter the buildings
    gdf = gdf[(gdf["gebruiksdoel"] != "") | (gdf.geometry.area > 30)]

    # Save the filtered GeoDataFrame
    gdf.to_file(output_file, driver="GeoJSON")

    print(f"Filtered buildings saved to {output_file}")


def create_mask(geojson_file: Path, mask_file: Path, bbox: BboxInt):
    print(f"Creating mask from {geojson_file}...")

    if mask_file.exists():
        print("File already exists.")
        return

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
    temp_file = mask_file.parent / "temp.tiff"
    mask_file.parent.mkdir(parents=True, exist_ok=True)

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
    clip_image(temp_file, mask_file, bbox)

    # Remove the temporary file
    temp_file.unlink()

    print(f"Rasterized mask saved to {mask_file}")


def tile_tiff_rasterio(image_path: Path, output_folder: Path, tile_size=512):
    print(f"Creating tiles from {image_path}...")

    if output_folder.exists():
        print("Folder already exists.")
        return

    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

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
                tile_path = output_folder / f"tile_{tile_index}.tif"
                with rasterio.open(tile_path, "w", **tile_meta) as dest:
                    dest.write(tile)

                tile_index += 1

    print(f"Saved {tile_index} tiles in {output_folder}")


def merge_tiff_files(
    input_files: list[Path], output_file: Path, nodata: float | None = None
):
    src_files_to_mosaic = [rasterio.open(f, "r") for f in input_files]
    nodata = nodata or src_files_to_mosaic[0].nodata
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata)

    # Define output metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "nodata": nodata,
        }
    )

    # Write the merged raster to a new file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close the source files
    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster saved as {output_file}")


def download_all(
    bbox: BboxInt, main_data_folder: Path, tile_size: int, filter_buildings=True
):
    data_folder = main_data_folder / bbox.folder_name()
    data_folder.mkdir(parents=True, exist_ok=True)

    # Save the bounding box to a file
    bbox_file = data_folder / "bbox.geojson"
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox.to_shapely()], crs="EPSG:28992")
    bbox_gdf.to_file(bbox_file, driver="GeoJSON")

    # Download the DSM
    dsm_file = data_folder / "dsm" / "merged.tif"
    download_dsm(dsm_file, bbox, main_data_folder)

    # Download the BAG buildings
    buildings_file = data_folder / "buildings" / "buildings.geojson"
    download_buildings(buildings_file, bbox)
    if filter_buildings:
        filtered_buildings_file = (
            data_folder / "buildings" / "buildings_filtered.geojson"
        )
        filter_small_buildings(buildings_file, filtered_buildings_file)
        buildings_file = filtered_buildings_file

    # Create a mask from the buildings
    mask_file = data_folder / "mask" / "merged.tif"
    if filter_buildings:
        mask_file = data_folder / "mask" / "merged_filtered.tif"
    create_mask(buildings_file, mask_file, bbox)

    # Create tiles from the DSM and mask
    tiles_dsm_folder = data_folder / "dsm" / f"tiles_{tile_size}"
    tiles_mask_folder = data_folder / "mask" / f"tiles_{tile_size}"
    if filter_buildings:
        tiles_mask_folder = data_folder / "mask" / f"tiles_{tile_size}_filtered"
    tile_tiff_rasterio(dsm_file, tiles_dsm_folder, tile_size=tile_size)
    tile_tiff_rasterio(mask_file, tiles_mask_folder, tile_size=tile_size)

    return data_folder, tiles_dsm_folder, tiles_mask_folder


if __name__ == "__main__":
    # minx, maxy, maxx, miny = 120000.0, 481500.0, 125000.0, 481250.0
    minx, maxy, maxx, miny = 80000, 457000, 85000, 452000
    main_data_folder = Path("../data")
    tile_size = 512

    bbox = (minx, maxy, maxx, miny)

    # download_all(bbox, main_data_folder, tile_size)
