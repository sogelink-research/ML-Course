import json
import warnings
import zipfile
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
from rasterio.warp import Resampling, calculate_default_transform, reproject
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
    # regions_url = "https://ns_hwh.fundaments.nl/hwh-ahn/AUX/bladwijzer.gpkg"
    regions_url = "https://basisdata.nl/hwh-ahn/AUX/bladwijzer.gpkg"
    regions_file = main_data_folder / "bladwijzer.gpkg"

    if not regions_file.exists():
        regions_file.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        response = requests.get(regions_url, stream=True, verify=False, headers=headers)
        response.raise_for_status()

        # Save the file
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
    for index, row in tqdm(intersecting_regions.iterrows()):
        if row["AHN5_05M_R"] is not None:
            ahn_urls.append(row["AHN5_05M_R"])
        else:
            ahn_urls.append(row["AHN4_05M_R"])

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
        The folder where the download_dsmdata is stored.

    Returns
    -------
    None
    """
    print("Downloading AHN5 DSM 50cm...", end=" ", flush=True)
    if tif_file.exists():
        print("Skipped because file already exists.")
        return

    urls = find_dsm_urls_for_bbox(main_data_folder, bbox)

    # Download the files
    temp_files = []
    tif_file.parent.mkdir(parents=True, exist_ok=True)

    for i, url in tqdm(enumerate(urls)):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        response = requests.get(url, stream=True, verify=False, headers=headers)
        response.raise_for_status()

        temp_file = tif_file.parent / url.split("/")[-1]
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Unzip the file if it is a zip file
        if temp_file.suffix == ".zip":
            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                zip_ref.extractall(temp_file.parent)
            temp_file.unlink()  # Remove the zip file
            temp_file = temp_file.parent / (str(temp_file.stem) + ".TIF")

        # Clip the image to the bbox
        clip_image(temp_file, temp_file, bbox)

        temp_files.append(temp_file)

    # Merge the temporary files into a single file
    merge_tiff_files(temp_files, tif_file, nodata=-9999)

    # Remove the temporary files
    for temp_file in temp_files:
        temp_file.unlink()

    print(f"Saved to {tif_file}.")


def clip_image(input_file: Path, output_file: Path, bbox: BboxInt):
    bbox_shapely = bbox.to_shapely()
    print(
        f"Clipping {input_file} to bbox {bbox_shapely.bounds}...", end=" ", flush=True
    )
    with rasterio.open(input_file, driver="GTiff") as src:
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
    print("Downloading BAG buildings...", end=" ", flush=True)

    if geojson_file.exists():
        print("Skipped because file already exists.")
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

    for sub_bbox in tqdm(bboxes):
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
    if len(all_features) == 0:
        buildings_gdf = gpd.GeoDataFrame(geometry=[])
    else:
        buildings_gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:28992")

    # Save the GeoDataFrame to a GeoJSON file
    geojson_file.parent.mkdir(parents=True, exist_ok=True)
    buildings_gdf.to_file(geojson_file, driver="GeoJSON")

    print(f"Saved to {geojson_file}.")


def filter_small_buildings(input_file: Path, output_file: Path):
    """
    Filter out the buildings that have the property "gebruiksdoel" set to NULL
    and an area smaller than 30 m².
    """
    print("Filtering buildings...", end=" ", flush=True)

    if output_file.exists():
        print("Skipped because file already exists.")
        return

    # Load the GeoDataFrame
    gdf = gpd.read_file(input_file)

    # Filter the buildings
    if not gdf.empty:
        gdf = gdf[(gdf["gebruiksdoel"] != "") | (gdf.geometry.area > 30)]

    # Save the filtered GeoDataFrame
    output_file.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_file, driver="GeoJSON")

    print(f"Saved to {output_file}.")


def create_mask(geojson_file: Path, mask_file: Path, bbox: BboxInt):
    print(f"Creating mask from {geojson_file}...", end=" ", flush=True)

    if mask_file.exists():
        print("Skipped because file already exists.")
        return

    # Load the vector data from GeoJSON file
    gdf = gpd.read_file(geojson_file)

    # Define the raster properties
    pixel_size = 0.5

    # Get the bounds of the vector data
    # minx, miny, maxx, maxy = gdf.total_bounds
    # if gdf.empty:
    minx, miny, maxx, maxy = bbox.to_float().to_tuple()

    print(f"Bounds: {minx}, {miny}, {maxx}, {maxy}")

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

    print(f"Saved to {mask_file}.")


def tile_tiff_rasterio(image_path: Path, output_folder: Path, tile_size=512):
    print(f"Creating tiles from {image_path}...", end=" ", flush=True)

    if output_folder.exists():
        print("Skipped because folder already exists.")
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

    print(f"Saved {tile_index} tiles in {output_folder}.")


def reproject_to_common_crs(input_file: Path, output_file: Path, target_crs: str):
    """
    Reproject a raster to a common CRS.

    Parameters
    ----------
    input_file : Path
        The path to the input raster file.
    output_file : Path
        The path to the output reprojected raster file.
    target_crs : str
        The target CRS (e.g., "EPSG:28992").
    """
    with rasterio.open(input_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(output_file, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )


def merge_tiff_files(
    input_files: list[Path], output_file: Path, nodata: float | None = None
):
    """
    Merge multiple raster files into one, ensuring they have a common CRS.

    Parameters
    ----------
    input_files : list[Path]
        List of input raster files.
    output_file : Path
        Path to the output merged raster file.
    nodata : float | None
        NoData value for the output raster.
    """
    target_crs = "EPSG:28992"

    if len(input_files) == 0:
        print("Cannot merge: no input files provided.")
        return

    # Reproject all input files to the target CRS
    for input_file in input_files:
        reproject_to_common_crs(input_file, input_file, target_crs)

    # Merge the reprojected files
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


def download_all(bbox: BboxInt, main_data_folder: Path, filter_buildings=True):
    """
    Download all the data for the given bounding box.
    This includes the DSM, the buildings, and the mask.
    The data is saved in a folder named after the bounding box.

    Parameters
    ----------
    bbox : BboxInt
        The bounding box.
    main_data_folder : Path
        The folder where the data is stored.
    filter_buildings : bool
        Whether to filter the buildings or not.

    Returns
    -------
    data_folder : Path
        The folder where the data is stored.
    dsm_file : Path
        The path to the DSM file.
    mask_file : Path
        The path to the mask file.
    """
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
            data_folder / "buildings_filtered" / "buildings.geojson"
        )
        filter_small_buildings(buildings_file, filtered_buildings_file)
        buildings_file = filtered_buildings_file

    # Create a mask from the buildings
    mask_file = data_folder / "mask" / "merged.tif"
    if filter_buildings:
        mask_file = data_folder / "mask_filtered" / "merged.tif"
    create_mask(buildings_file, mask_file, bbox)

    return data_folder, dsm_file, mask_file


def tile_image(image_file: Path, tile_size: int) -> Path:
    """
    Tile the image into smaller images of the given size.

    Parameters
    ----------
    image_file : Path
        The path to the image file.
    tile_size : int
        The size of the tiles.

    Returns
    -------
    Path
        The path to the folder where the tiles are saved.
    """
    output_folder = image_file.parent / f"tiles_{tile_size}"
    tile_tiff_rasterio(image_file, output_folder, tile_size=tile_size)
    return output_folder
