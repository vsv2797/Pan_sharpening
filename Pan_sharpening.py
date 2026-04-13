import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box

# =============================================================================
# 1. INPUT PATHS
# =============================================================================
# --- Landsat 8 L1TP individual band TIFs ---
LANDSAT_DIR = "data/Landsat"
LS_PREFIX   = "LC08_L1TP_014032_20260317_20260406_02_T1"

landsat_pan_path = f"{LANDSAT_DIR}/{LS_PREFIX}_B8.TIF"   # Panchromatic  15 m
landsat_b2_path  = f"{LANDSAT_DIR}/{LS_PREFIX}_B2.TIF"   # Blue          30 m
landsat_b3_path  = f"{LANDSAT_DIR}/{LS_PREFIX}_B3.TIF"   # Green         30 m
landsat_b4_path  = f"{LANDSAT_DIR}/{LS_PREFIX}_B4.TIF"   # Red           30 m

# --- Sentinel-2 L2A jp2 files (10 m RGB + NIR) ---
S2_R10M = (
    "data/sentinel/GRANULE/"
    "L2A_T18TWL_A056379_20260408T155034/"
    "IMG_DATA/R10m"
)
S2_PREFIX = "T18TWL_20260408T154711"

sentinel_b02_path = f"{S2_R10M}/{S2_PREFIX}_B02_10m.jp2"  # Blue   10 m
sentinel_b03_path = f"{S2_R10M}/{S2_PREFIX}_B03_10m.jp2"  # Green  10 m
sentinel_b04_path = f"{S2_R10M}/{S2_PREFIX}_B04_10m.jp2"  # Red    10 m

# --- AOI: Central Park (WGS84) ---
# left, bottom, right, top
central_park_bbox = (-73.9819, 40.7642, -73.9498, 40.8007)
aoi = gpd.GeoDataFrame(
    {"name": ["Central Park"]},
    geometry=[box(*central_park_bbox)],
    crs="EPSG:4326",
)

# =============================================================================
# 2. LOAD RASTERS
# =============================================================================
# Landsat bands (each single-band TIF -> shape [1, H, W])
pan  = rxr.open_rasterio(landsat_pan_path, masked=True)
ls_b2 = rxr.open_rasterio(landsat_b2_path, masked=True)
ls_b3 = rxr.open_rasterio(landsat_b3_path, masked=True)
ls_b4 = rxr.open_rasterio(landsat_b4_path, masked=True)

# Sentinel-2 individual bands
s2_b02 = rxr.open_rasterio(sentinel_b02_path, masked=True)
s2_b03 = rxr.open_rasterio(sentinel_b03_path, masked=True)
s2_b04 = rxr.open_rasterio(sentinel_b04_path, masked=True)

print("=== LANDSAT PANCHROMATIC (B8) ===")
print("  CRS:", pan.rio.crs)
print("  Bounds:", pan.rio.bounds())
print("  Resolution:", pan.rio.resolution())
print("  Shape:", pan.shape)

print("\n=== LANDSAT MULTISPECTRAL (B2/B3/B4) ===")
print("  CRS:", ls_b2.rio.crs)
print("  Bounds:", ls_b2.rio.bounds())
print("  Resolution:", ls_b2.rio.resolution())
print("  Shape:", ls_b2.shape)

print("\n=== SENTINEL-2 B02 (10 m) ===")
print("  CRS:", s2_b02.rio.crs)
print("  Bounds:", s2_b02.rio.bounds())
print("  Resolution:", s2_b02.rio.resolution())
print("  Shape:", s2_b02.shape)

# =============================================================================
# 3. REPROJECT AOI TO EACH RASTER CRS AND CLIP
# =============================================================================
def clip_to_aoi(da, aoi_wgs84):
    """Reproject AOI to raster CRS, then clip."""
    aoi_local = aoi_wgs84.to_crs(da.rio.crs)
    return da.rio.clip(aoi_local.geometry, aoi_local.crs, drop=True)

pan_clip  = clip_to_aoi(pan,   aoi)
b2_clip   = clip_to_aoi(ls_b2, aoi)
b3_clip   = clip_to_aoi(ls_b3, aoi)
b4_clip   = clip_to_aoi(ls_b4, aoi)

s2_b02_clip = clip_to_aoi(s2_b02, aoi)
s2_b03_clip = clip_to_aoi(s2_b03, aoi)
s2_b04_clip = clip_to_aoi(s2_b04, aoi)

print("\n=== CLIPPED PAN ===", pan_clip.shape)
print("=== CLIPPED LS MS ===", b2_clip.shape)
print("=== CLIPPED S2    ===", s2_b02_clip.shape)

# =============================================================================
# 4. PAN-SHARPENING (Brovey Transform)
#    Upsample 30 m MS to 15 m, then apply Brovey:
#    band_sharp = band / (R + G + B) * PAN
# =============================================================================
# Upsample multispectral bands to match the PAN grid
b2_up = b2_clip.rio.reproject_match(pan_clip)
b3_up = b3_clip.rio.reproject_match(pan_clip)
b4_up = b4_clip.rio.reproject_match(pan_clip)

# Convert to float32
pan_f = pan_clip.values[0].astype("float32")
r_f   = b4_up.values[0].astype("float32")   # Red   = B4
g_f   = b3_up.values[0].astype("float32")   # Green = B3
b_f   = b2_up.values[0].astype("float32")   # Blue  = B2

total = r_f + g_f + b_f + 1e-6  # avoid div/0

r_sharp = np.clip(r_f / total * pan_f, 0, None)
g_sharp = np.clip(g_f / total * pan_f, 0, None)
b_sharp = np.clip(b_f / total * pan_f, 0, None)

pan_sharpened_rgb = np.stack([r_sharp, g_sharp, b_sharp], axis=0)  # [3, H, W]

print("\n=== PAN-SHARPENED OUTPUT ===")
print("  Shape:", pan_sharpened_rgb.shape)

# =============================================================================
# 5. NORMALISE FOR DISPLAY
# =============================================================================
def normalize_rgb(arr_chw):
    """Percentile-stretch each channel; return HWC float32 in [0,1]."""
    arr = arr_chw.astype("float32")
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        band = arr[i]
        p2, p98 = np.nanpercentile(band[np.isfinite(band)], (2, 98))
        out[i] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    return np.transpose(out, (1, 2, 0))  # -> HWC

# Original 30 m multispectral (B4, B3, B2 = RGB)
ls_ms_rgb_chw = np.stack(
    [b4_clip.values[0], b3_clip.values[0], b2_clip.values[0]], axis=0
).astype("float32")

# Sentinel-2 10 m reference (B04, B03, B02 = RGB)
s2_rgb_chw = np.stack(
    [s2_b04_clip.values[0], s2_b03_clip.values[0], s2_b02_clip.values[0]], axis=0
).astype("float32")

# Resample Sentinel-2 to PAN grid for direct comparison
s2_b04_match = s2_b04_clip.rio.reproject_match(pan_clip)
s2_b03_match = s2_b03_clip.rio.reproject_match(pan_clip)
s2_b02_match = s2_b02_clip.rio.reproject_match(pan_clip)
s2_rgb_matched_chw = np.stack(
    [s2_b04_match.values[0], s2_b03_match.values[0], s2_b02_match.values[0]], axis=0
).astype("float32")

ls_ms_disp       = normalize_rgb(ls_ms_rgb_chw)
pan_sharp_disp   = normalize_rgb(pan_sharpened_rgb)
s2_disp          = normalize_rgb(s2_rgb_matched_chw)

# =============================================================================
# 6. VISUALISE: Original MS | Pan-Sharpened | Sentinel-2 Reference
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

axes[0].imshow(ls_ms_disp)
axes[0].set_title("Landsat MS (30 m) — stretched to PAN grid")
axes[0].axis("off")

axes[1].imshow(pan_sharp_disp)
axes[1].set_title("Pan-Sharpened Landsat (Brovey, 15 m)")
axes[1].axis("off")

axes[2].imshow(s2_disp)
axes[2].set_title("Sentinel-2 reference (10 m, resampled to 15 m grid)")
axes[2].axis("off")

plt.suptitle("Pan-Sharpening: Central Park — Landsat 8 × Sentinel-2", fontsize=13)
plt.tight_layout()
plt.show()

# =============================================================================
# 7. EDGE-BASED QUALITY CHECK
# =============================================================================
def edge_mag(gray):
    gy, gx = np.gradient(gray)
    return np.sqrt(gx**2 + gy**2)

ps_gray  = np.nanmean(pan_sharp_disp, axis=2)
s2_gray  = np.nanmean(s2_disp, axis=2)
ms_gray  = normalize_rgb(ls_ms_rgb_chw)
ms_gray  = np.nanmean(ms_gray, axis=2)

# Resize MS display to PAN grid via simple repeat (rough, for metric only)
ps_edge  = edge_mag(ps_gray)
s2_edge  = edge_mag(s2_gray)

valid = np.isfinite(ps_edge) & np.isfinite(s2_edge)
corr  = np.corrcoef(ps_edge[valid].ravel(), s2_edge[valid].ravel())[0, 1]

print(f"\nEdge correlation (pan-sharpened vs Sentinel-2): {corr:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(ps_edge, cmap="gray")
axes[0].set_title("Pan-Sharpened edges")
axes[0].axis("off")

axes[1].imshow(s2_edge, cmap="gray")
axes[1].set_title("Sentinel-2 edges (reference)")
axes[1].axis("off")

plt.tight_layout()
plt.show()