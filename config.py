"""
config.py — Central configuration for the co-registration pipeline.

Edit REFERENCE_GLOB and TARGET_GLOB to point to your data.
All other parameters are shared across every script in the project.
"""

# ── Data paths ─────────────────────────────────────────────────────────────────
# Glob patterns that match ALL tiles of the reference and target mosaics.
REFERENCE_GLOB = r"C:\Users\LENOVO\Rainforest Builder Dropbox\03_RB Ghana - All Team\03_Planning\00_Spatial\04_Drone Imagery\01_Data\01_Forest Reserves\03_Anhwiaso South Compartment\2026\ASO_S13-03-26_E16-03-26_R48\*.tif"
TARGET_GLOB    = r"C:\Users\LENOVO\Rainforest Builder Dropbox\03_RB Ghana - All Team\03_Planning\00_Spatial\04_Drone Imagery\01_Data\01_Forest Reserves\03_Anhwiaso South Compartment\2026\ASO_S21-04-26_E21-04-26_R53\*.tif"

# ── Tiling and detection ────────────────────────────────────────────────────────
TILE_PX   = 8192   # geographic grid cell size (pixels)
DETECT_PX = 2048   # SIFT detection resolution — smaller = faster and less memory
BAND      = 1      # raster band used for SIFT (1 = Red for RGBA drone imagery)
N_CPUS    = 4      # parallel worker processes

# ── Quality filters ─────────────────────────────────────────────────────────────
MAD_K        = 3.0  # cells whose shift is > MAD_K × MAD from the median are excluded
                    # from the consensus before computing the fallback median
MAX_SHIFT_PX = 20   # per-match displacement limit (detection pixels, ≈ 5 m at typical
                    # drone resolution) — RANSAC inliers beyond this are discarded
