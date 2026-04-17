##Enhancement of Spatial Resolution of Low-Resolution Multispectral Satellite Imagery Using Pan Sharpening.

A Python-based geospatial workflow for preprocessing, visualizing, and checking co-registration between remote sensing images over a selected area of interest. This project is focused on building a clean foundation for machine learning and Earth observation workflows using **Landsat 8** and **Sentinel-2** data.

## Overview

This repository contains scripts and notebooks for:
- loading satellite imagery,
- clipping data to an AOI,
- visualizing multispectral images,
- comparing image alignment,
- estimating pixel shifts between datasets,
- preparing data for later machine learning or fusion tasks.

The current AOI for this project is **Central Park, New York**.

## Why this project?

Remote sensing and machine learning pipelines are only as good as their preprocessing. Before training a model or doing image fusion, the input rasters must be properly aligned, clipped to the same region, and checked for spatial consistency. This project provides a practical workflow for doing exactly that.

## Study Area

**Area of Interest:** Central Park, New York City  
The AOI was selected because it is a compact urban environment with:
- strong linear features,
- mixed land cover,
- clear spatial boundaries,
- and useful structures for visual co-registration checks.

## Data Sources

This project is designed to work with:
- **Landsat 8** imagery,
- **Sentinel-2** imagery.

The script assumes that the rasters have already been downloaded locally and are available in the expected folder structure.

## Workflow

The pipeline follows these steps:

1. Load Landsat 8 and Sentinel-2 raster data.
2. Define or import the Central Park AOI.
3. Reproject the AOI to match each raster CRS.
4. Clip both images to the AOI.
5. Match one raster to the other’s grid.
6. Visualize both images side by side.
7. Overlay the images for a quick alignment check.
8. Estimate pixel shift using a correlation-based approach.
9. Prepare the processed outputs for the next stage of the project.

## Project Structure

```text
project-name/
├── data/
│   ├── landsat/
│   ├── sentinel/
│   └── aoi/
├── notebooks/
├── scripts/
├── outputs/
│   ├── figures/
│   └── processed/
└── README.md
```

## Requirements

This project uses Python geospatial libraries such as:
- `rasterio`
- `rioxarray`
- `xarray`
- `geopandas`
- `numpy`
- `matplotlib`

You can install them with:

```bash
pip install rasterio rioxarray xarray geopandas numpy matplotlib
```

## How to Use

1. Place your Landsat and Sentinel-2 files in the `data/` directory.
2. Update the file paths in the processing script.
3. Run the script to clip, align, and visualize the data.
4. Review the output figures and pixel shift estimates.
5. Use the processed rasters for further analysis or ML model development.

## Results

This project produces:
- cropped AOI rasters,
- alignment visualizations,
- overlay comparisons,
- pixel shift estimates,
- and processed outputs ready for downstream work.

## Image Placeholders

Add your own figures here once you run the pipeline.

### AOI map
![AOI map](images/aoi-map.png)

### Landsat image
![Landsat image](images/landsat-central-park.png)

### Sentinel-2 image
![Sentinel-2 image](images/sentinel-central-park.png)

### Overlay comparison
![Overlay comparison](images/overlay-comparison.png)

### Co-registration check
![Co-registration check](images/coregistration-check.png)

## Example Output

A successful run should show:
- clear cropping to the Central Park boundary,
- visible but minimal spatial difference between the images,
- strong overlap in major edges and structures,
- and a small estimated pixel shift.

## Next Steps

Planned improvements for this project include:
- proper reflectance scaling,
- band-specific visualization,
- automatic registration metrics,
- fusion or pan-sharpening experiments,
- and preparation of training data for deep learning models.

## Notes

This repository is being developed as part of a remote sensing and machine learning learning path. The first goal is to build a reliable preprocessing pipeline before moving to model training.

## License

Add your preferred license here.
