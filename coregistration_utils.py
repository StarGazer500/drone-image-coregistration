"""
image_coregistration_tiled.py — Shared utilities for per-tile co-registration.

Both main_homography_tiled.py and main_affine_tiled.py import from here.

Core idea
---------
Instead of matching irregular source tiles against each other, both
reference and target mosaics are read through their VRTs and sliced into
a *uniform grid* of equal-sized cells.  Every cell pair (ref_cell_i,
tgt_cell_i) covers exactly the same geographic bounds at the same pixel
dimensions, so SIFT has a fair, consistent input regardless of how the
original tiles were laid out.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds as _wfb
from rasterio.transform import Affine
from rasterio.enums import Resampling


# ── VRT builder ───────────────────────────────────────────────────────────────

def build_vrt(tile_list, vrt_path):
    """Build a GDAL VRT mosaic from a list of tile paths."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.BuildVRT(vrt_path, sorted(tile_list))
    if ds is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {vrt_path}")
    ds.FlushCache()
    ds = None
    return vrt_path


# ── Tile grid ─────────────────────────────────────────────────────────────────

def compute_grid(tgt_vrt_path, tile_px=4096):
    """
    Partition the target VRT's full extent into a uniform grid of cells.

    Every cell is at most (tile_px × tile_px) pixels.  Edge cells are
    smaller so the grid exactly covers the mosaic.

    Returns a list of tuples:
        (row, col, left, bottom, right, top, w_px, h_px, res)
    """
    with rasterio.open(tgt_vrt_path) as ds:
        b   = ds.bounds
        res = abs(ds.transform.a)

    tile_map = tile_px * res
    n_cols   = int(np.ceil((b.right - b.left)   / tile_map))
    n_rows   = int(np.ceil((b.top   - b.bottom) / tile_map))

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            left   = b.left + c * tile_map
            top    = b.top  - r * tile_map
            right  = min(left + tile_map, b.right)
            bottom = max(top  - tile_map, b.bottom)
            w_px   = max(1, round((right  - left)   / res))
            h_px   = max(1, round((top    - bottom) / res))
            cells.append((r, c, left, bottom, right, top, w_px, h_px, res))

    return cells


# ── Raster I/O ────────────────────────────────────────────────────────────────

def read_tile_band(vrt_path, left, bottom, right, top, h_px, w_px, band):
    """
    Read one band from a VRT at the given map bounds into a float32 array
    of shape (h_px, w_px).  Regions outside the VRT coverage are zeros.
    """
    out = np.zeros((h_px, w_px), dtype=np.float32)
    with rasterio.open(vrt_path) as src:
        sb = src.bounds
        il = max(left,   sb.left);   ir = min(right,  sb.right)
        ib = max(bottom, sb.bottom); it = min(top,    sb.top)
        if ir <= il or it <= ib:
            return out

        # Output-pixel size
        px_w = (right - left)   / w_px
        px_h = (top   - bottom) / h_px

        # Offset and size of the overlap region in output-pixel space
        ox = max(0, round((il - left) / px_w))
        oy = max(0, round((top - it)  / px_h))
        sw = min(w_px - ox, max(1, round((ir - il) / px_w)))
        sh = min(h_px - oy, max(1, round((it - ib) / px_h)))
        if sw <= 0 or sh <= 0:
            return out

        win = _wfb(il, ib, ir, it, src.transform)
        b   = min(band, src.count)
        arr = src.read(b, window=win, out_shape=(sh, sw),
                       resampling=Resampling.bilinear)
        out[oy:oy + sh, ox:ox + sw] = arr.astype(np.float32)

    return out


def read_tile_all(vrt_path, left, bottom, right, top, h_px, w_px):
    """
    Read ALL bands from a VRT at the given map bounds into an array of
    shape (n_bands, h_px, w_px).  Regions outside VRT coverage are zeros.
    """
    with rasterio.open(vrt_path) as src:
        n_bands  = src.count
        out_dtype = src.dtypes[0]
        sb = src.bounds
        il = max(left,   sb.left);   ir = min(right,  sb.right)
        ib = max(bottom, sb.bottom); it = min(top,    sb.top)

        out = np.zeros((n_bands, h_px, w_px), dtype=out_dtype)
        if ir <= il or it <= ib:
            return out

        px_w = (right - left)   / w_px
        px_h = (top   - bottom) / h_px
        ox   = max(0, round((il - left) / px_w))
        oy   = max(0, round((top - it)  / px_h))
        sw   = min(w_px - ox, max(1, round((ir - il) / px_w)))
        sh   = min(h_px - oy, max(1, round((it - ib) / px_h)))
        if sw <= 0 or sh <= 0:
            return out

        win = _wfb(il, ib, ir, it, src.transform)
        sub = src.read(window=win, out_shape=(n_bands, sh, sw),
                       resampling=Resampling.bilinear)
        out[:, oy:oy + sh, ox:ox + sw] = sub

    return out


# ── SIFT prep ─────────────────────────────────────────────────────────────────

def to_uint8(arr):
    """Normalise any-dtype 2-D array to uint8 for SIFT."""
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return np.zeros(arr.shape, np.uint8)
    return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


def histogram_match(ref_u8, tgt_u8, n_bins=1024):
    """In-memory histogram match of tgt_u8 to ref_u8 (both 2-D uint8)."""
    r  = ref_u8.astype(np.float32)
    t  = tgt_u8.astype(np.float32)
    rv = r[r > 0]
    tv = t[t > 0]
    if rv.size < 50 or tv.size < 50:
        return tgt_u8
    lo = float(min(rv.min(), tv.min()))
    hi = float(max(rv.max(), tv.max()))
    if lo >= hi:
        return tgt_u8
    rh, edges = np.histogram(rv, bins=n_bins, range=(lo, hi))
    th, _     = np.histogram(tv, bins=n_bins, range=(lo, hi))
    rc = np.cumsum(rh).astype(np.float64); rc /= rc[-1]
    tc = np.cumsum(th).astype(np.float64); tc /= tc[-1]
    lut  = np.interp(tc, rc, 0.5 * (edges[:-1] + edges[1:]))
    out  = t.copy()
    mask = t > 0
    out[mask] = lut[np.clip(np.searchsorted(edges[1:], t[mask]), 0, n_bins - 1)]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Corrected-tile writers ────────────────────────────────────────────────────

def write_translation_tile(tgt_vrt, left, bottom, right, top, h_px, w_px,
                            dx_map, dy_map, res, out_path):
    """
    Read all target bands at the given cell bounds, shift the GeoTransform
    origin by (dx_map, dy_map), and write a compressed GeoTIFF.
    No pixel resampling — pixels are identical to the source.
    """
    with rasterio.open(tgt_vrt) as src:
        nodata = src.nodata
        crs    = src.crs
        dtype  = src.dtypes[0]

    data = read_tile_all(tgt_vrt, left, bottom, right, top, h_px, w_px)

    meta = {
        "driver":    "GTiff",
        "dtype":     dtype,
        "nodata":    nodata,
        "width":     w_px,
        "height":    h_px,
        "count":     data.shape[0],
        "crs":       crs,
        "transform": Affine(res, 0, left + dx_map, 0, -res, top + dy_map),
        "compress":  "lzw",
        "predictor": 2,
        "BIGTIFF":   "IF_SAFER",
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    # Cast to source dtype (read_tile_all may return float for edge tiles)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(np.dtype(dtype))
        data = np.clip(data.astype(np.float32), info.min, info.max)
    data = data.astype(dtype)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)
    return out_path


def write_affine_tile(tgt_vrt, left, bottom, right, top, h_px, w_px,
                       ref_map_pts, tgt_px_pts, srs_wkt, res, out_path):
    """
    Extract the target tile from the VRT, attach GCPs derived from SIFT
    inliers, and apply a 1st-order polynomial (similarity) warp via GDAL.
    Output is reprojected back to the same cell bounds at resolution `res`.
    """
    from osgeo import gdal
    gdal.UseExceptions()

    # ── 1. Extract raw target tile as a temp GTiff ─────────────────────────
    tmp_src = out_path + "_raw.tif"
    ds = gdal.Translate(
        tmp_src, tgt_vrt,
        projWin=(left, top, right, bottom),
        xRes=res, yRes=res,
        format="GTiff",
    )
    if ds is None:
        raise RuntimeError(f"gdal.Translate failed extracting tile for {out_path}")
    ds.FlushCache(); ds = None

    # ── 2. Attach GCPs to a VRT copy ──────────────────────────────────────
    tmp_vrt = out_path + "_gcp.vrt"
    src_ds  = gdal.Open(tmp_src)
    vrt_ds  = gdal.GetDriverByName("VRT").CreateCopy(tmp_vrt, src_ds)
    gcps    = [
        gdal.GCP(
            float(ref_map_pts[i, 0]),  # GCPX  = reference longitude
            float(ref_map_pts[i, 1]),  # GCPY  = reference latitude
            0.0,
            float(tgt_px_pts[i, 0]),   # GCPPixel = target tile column
            float(tgt_px_pts[i, 1]),   # GCPLine  = target tile row
        )
        for i in range(len(ref_map_pts))
    ]
    vrt_ds.SetGCPs(gcps, srs_wkt)
    vrt_ds.FlushCache()
    vrt_ds = src_ds = None

    # ── 3. Warp back to the cell's exact extent ────────────────────────────
    opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(left, bottom, right, top),
        xRes=res, yRes=res,
        polynomialOrder=1,
        creationOptions=[
            "COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=IF_SAFER",
            "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
            "NUM_THREADS=ALL_CPUS",
        ],
    )
    out_ds = gdal.Warp(out_path, tmp_vrt, options=opts)
    if out_ds is None:
        os.remove(tmp_src); os.remove(tmp_vrt)
        raise RuntimeError(f"gdal.Warp failed for {out_path}")
    out_ds.FlushCache(); out_ds = None

    os.remove(tmp_src)
    os.remove(tmp_vrt)
    return out_path


def write_affine_geotransform_tile(tgt_vrt, left, bottom, right, top, h_px, w_px,
                                    M, det_w, det_h, res, out_path):
    """
    Apply the full affine M (translation + rotation + scale) by encoding it
    directly into the output GeoTransform.  No pixel resampling — pixels are
    written as-is, identical to the source.

    How it works
    ------------
    For target full-resolution pixel (col, row), the corrected map coordinate is:

        lon = M[0,0]*res*col + M[0,1]*res*row + (left + M[0,2]*det_px_x)
        lat =-M[1,0]*res*col - M[1,1]*res*row + (top  - M[1,2]*det_px_y)

    This maps directly to rasterio Affine(a, b, c, d, e, f):
        a = M[0,0]*res      b = M[0,1]*res      c = left + M[0,2]*det_px_x
        d =-M[1,0]*res      e =-M[1,1]*res      f = top  - M[1,2]*det_px_y

    For identity M this reduces to the standard Affine(res, 0, left, 0, -res, top).

    Quality vs write_affine_matrix_tile (GCP warp)
    -----------------------------------------------
    GCP warp: resamples pixels per tile → interpolation artifacts + edge risk.
    This function: zero resampling per tile; one resampling when mosaic_cog
    reprojects rotated tiles to a north-up COG (unavoidable, done once only).
    """
    det_px_x = (right - left)   / det_w
    det_px_y = (top   - bottom) / det_h

    corrected_transform = Affine(
         float(M[0, 0]) * res,                  # a  x-scale (with rotation+scale)
         float(M[0, 1]) * res,                  # b  x from row (rotation term)
         left + float(M[0, 2]) * det_px_x,      # c  x origin  (translation)
        -float(M[1, 0]) * res,                  # d  y from col (rotation term)
        -float(M[1, 1]) * res,                  # e  y-scale (with rotation+scale)
         top  - float(M[1, 2]) * det_px_y,      # f  y origin  (translation)
    )

    with rasterio.open(tgt_vrt) as src:
        nodata = src.nodata
        crs    = src.crs
        dtype  = src.dtypes[0]

    data = read_tile_all(tgt_vrt, left, bottom, right, top, h_px, w_px)

    meta = {
        "driver":    "GTiff",
        "dtype":     dtype,
        "nodata":    nodata,
        "width":     w_px,
        "height":    h_px,
        "count":     data.shape[0],
        "crs":       crs,
        "transform": corrected_transform,
        "compress":  "lzw",
        "predictor": 2,
        "BIGTIFF":   "IF_SAFER",
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(np.dtype(dtype))
        data = np.clip(data.astype(np.float32), info.min, info.max)
    data = data.astype(dtype)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)
    return out_path


def write_affine_matrix_tile(tgt_vrt, left, bottom, right, top, h_px, w_px,
                              M, det_w, det_h, srs_wkt, res, out_path, n_gcp=7):
    """
    Apply affine matrix M (from estimateAffinePartial2D) to a full-resolution
    target cell, correcting translation + rotation + scale.

    Uses a regular n×n synthetic GCP grid derived from M rather than sparse
    SIFT keypoint positions.  This avoids clustered/edge-gap GCPs and gives
    a well-conditioned polynomial fit across the full tile extent.

    M        : 2×3 float32 — maps target det-pixel → reference det-pixel
    det_w/h  : detection-resolution dimensions used when M was estimated
    n_gcp    : side length of GCP grid  (n_gcp² control points total)
    """
    from osgeo import gdal
    gdal.UseExceptions()

    # ── 1. Extract raw target tile ────────────────────────────────────────────
    tmp_src = out_path + "_raw.tif"
    ds = gdal.Translate(
        tmp_src, tgt_vrt,
        projWin=(left, top, right, bottom),
        xRes=res, yRes=res,
        format="GTiff",
    )
    if ds is None:
        raise RuntimeError(f"gdal.Translate failed extracting tile for {out_path}")
    ds.FlushCache(); ds = None

    # ── 2. Build synthetic GCP grid from M ───────────────────────────────────
    cols_det = np.linspace(0, det_w - 1, n_gcp)
    rows_det = np.linspace(0, det_h - 1, n_gcp)

    gcps = []
    for rd in rows_det:
        for cd in cols_det:
            ref_det     = M @ np.array([cd, rd, 1.0], dtype=np.float64)
            ref_lon     = left + ref_det[0] * (right - left) / det_w
            ref_lat     = top  - ref_det[1] * (top   - bottom) / det_h
            tgt_col_full = cd * w_px / det_w
            tgt_row_full = rd * h_px / det_h
            gcps.append(gdal.GCP(
                float(ref_lon), float(ref_lat), 0.0,
                float(tgt_col_full), float(tgt_row_full),
            ))

    # ── 3. Attach GCPs to a VRT copy ─────────────────────────────────────────
    tmp_vrt = out_path + "_gcp.vrt"
    src_ds  = gdal.Open(tmp_src)
    vrt_ds  = gdal.GetDriverByName("VRT").CreateCopy(tmp_vrt, src_ds)
    vrt_ds.SetGCPs(gcps, srs_wkt)
    vrt_ds.FlushCache()
    vrt_ds = src_ds = None

    # ── 4. Warp back to the cell's north-up grid ─────────────────────────────
    opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(left, bottom, right, top),
        xRes=res, yRes=res,
        polynomialOrder=1,
        creationOptions=[
            "COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=IF_SAFER",
            "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
            "NUM_THREADS=ALL_CPUS",
        ],
    )
    out_ds = gdal.Warp(out_path, tmp_vrt, options=opts)
    if out_ds is None:
        for p in (tmp_src, tmp_vrt):
            try: os.remove(p)
            except OSError: pass
        raise RuntimeError(f"gdal.Warp failed for {out_path}")
    out_ds.FlushCache(); out_ds = None

    os.remove(tmp_src)
    os.remove(tmp_vrt)
    return out_path


# ── Final mosaic ──────────────────────────────────────────────────────────────

def mosaic_cog(tile_paths, output_path, tmp_dir):
    """
    Mosaic a list of corrected tiles into a single Cloud-Optimized GeoTIFF.

    Uses gdal.Warp instead of BuildVRT + Translate so that tiles with rotated
    GeoTransforms (from write_affine_geotransform_tile) are included correctly.
    gdal.BuildVRT silently skips rotated inputs; gdal.Warp reprojects them to
    a north-up output grid automatically.
    """
    from osgeo import gdal
    gdal.UseExceptions()

    # Always use GTiff, not the COG driver.  The COG driver writes a full
    # uncompressed intermediate copy (~31 GB for a 126k×62k image) before
    # reordering into COG layout — this causes out-of-space errors.
    # GTiff with TILED+COMPRESS streams directly with no intermediate file.
    opts = gdal.WarpOptions(
        format="GTiff",
        creationOptions=[
            "COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=IF_SAFER",
            "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
            "NUM_THREADS=ALL_CPUS",
        ],
    )
    out = gdal.Warp(output_path, list(tile_paths), options=opts)
    if out is None:
        raise RuntimeError(f"gdal.Warp failed mosaicking to {output_path}")

    # Build overview pyramids after the warp (not during) so there is no
    # uncompressed intermediate file on disk.  Without these QGIS reads the
    # full-resolution raster when zoomed out, making large mosaics very slow.
    print("  Building overviews ...")
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "LZW")
    gdal.SetConfigOption("PREDICTOR_OVERVIEW", "2")
    out.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64])
    out.FlushCache()
    out = None

    return output_path
