"""
config.py — Central configuration for the co-registration pipeline.

Edit REFERENCE_GLOB and TARGET_GLOB to point to your data.
All other parameters are shared across every script in the project.
"""

# ── Data paths ─────────────────────────────────────────────────────────────────
# Glob patterns that match ALL tiles of the reference and target mosaics.
REFERENCE_GLOB = r"C:\Users\LENOVO\Rainforest Builder Dropbox\03_RB Ghana - All Team\03_Planning\00_Spatial\04_Drone Imagery\01_Data\01_Forest Reserves\03_Anhwiaso South Compartment\2026\ASO_S13-03-26_E16-03-26_R48\*.tif"
TARGET_GLOB    = r"C:\Users\LENOVO\Rainforest Builder Dropbox\03_RB Ghana - All Team\03_Planning\00_Spatial\04_Drone Imagery\01_Data\01_Forest Reserves\03_Anhwiaso South Compartment\2026\ASO_S21-05-26_E21-05-26_R53\*.tif"

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

# ── AROSICS local co-registration (run_arosics.py only) ─────────────────────────
AROSICS_GRID_RES  = 100          # tie-point grid spacing in pixels of the downsampled input
AROSICS_WIN_SIZE  = (512, 512)   # (cols, rows) NCC matching window in pixels
AROSICS_MAX_SHIFT = 50           # maximum expected shift in pixels
AROSICS_MAX_PX    = 4096         # max pixels along the longest edge before passing to AROSICS;
                                 # large mosaics are downsampled to this size to stay within RAM

# Correction strategy after shift detection:
#   "translation" — one constant shift per grid tile (zero pixel resampling;
#                   fast; accurate when shift varies slowly across the mosaic)
#   "spline"      — thin-plate spline warp applied per-pixel inside every tile
#                   (one bilinear resample; handles any spatial shift pattern;
#                   recommended for datasets with strongly non-uniform GPS drift)
AROSICS_CORRECTION    = "translation"  # "translation" or "spline"
                                       # translation: zero pixel resampling, minimal disk use
                                       # spline: per-pixel warp, needs ~2× disk space for tiles
AROSICS_SPLINE_EVAL_N = 16             # only used when AROSICS_CORRECTION = "spline"

