clip := "121114 485081 123548 483440"
bbox := "121114,485081,123548,483440"

@default:
	just -f dataparse.just --list

all: download_ahn download_buildings create_mask create_tiles

download_ahn:
	wget -nc "https://ns_hwh.fundaments.nl/hwh-ahn/AHN2/DSM_50cm/r25gn1.tif.zip"
	unzip -n r25gn1.tif.zip -d image.tiff

# create_image:
# 	gdal_translate r25gn1.tif image.tiff -projwin {{clip}}

download_buildings:
	image_extent_up_left := $(shell gdalinfo image.tiff | grep "Upper Left" | tr -d '[:space:]' | awk -F '[\(\)]' '{print $2}')
	image_extent_low_right := $(shell gdalinfo image.tiff | grep "Lower Right" | tr -d '[:space:]' | awk -F '[\(\)]' '{print $2}')
	image_extent := $(image_extent_up_left),$(image_extent_low_right)
	ogr2ogr panden.geojson \
		WFS:"https://service.pdok.nl/lv/bag/wfs/v2_0?request=getCapabilities&service=WFS&bbox={{image_extent}}" bag:pand \

create_mask:
	gdal_rasterize \
		-l bag:pand \
		-burn 1.0 \
		-tr 0.5 0.5 \
		-tap \
		-a_nodata 0.0 \
		-ot Float32 \
		-of GTIFF panden.geojson panden_mask.tiff
	image_extent_up_left := $(shell gdalinfo image.tiff | grep "Upper Left" | tr -d '[:space:]' | awk -F '[\(\),]' '{print $2" "$3}')
	image_extent_low_right := $(shell gdalinfo image.tiff | grep "Lower Right" | tr -d '[:space:]' | awk -F '[\(\),]' '{print $2" "$3}')
	image_extent := $(image_extent_up_left) $(image_extent_low_right)
	gdal_translate panden_mask.tiff mask.tiff -projwin {{image_extent}}

create_tiles:
	mkdir -p tiles_image
	mkdir -p tiles_mask
	gdal_retile.py -ps 512 512 -targetDir tiles_image image.tiff
	gdal_retile.py -ps 512 512 -targetDir tiles_mask mask.tiff
	for f in tiles_image/image_*.tif; do mv "$f" "${f/image_/}" done
	for f in tiles_mask/mask_*.tif; do mv "$f" "${f/mask_/}" done
	tar cvf tiles.tar.gz tiles_image tiles_mask

upload:
	gcloud storage cp -r tiles.tar.gz gs://sogelink-research-public/projects/tomt/ml_coursedata/tiles2.tar.gz
 