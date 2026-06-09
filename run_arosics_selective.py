"""
run_arosics.py — Local co-registration using AROSICS NCC shift detection.

Workflow
--------
1. Build VRTs from all reference and target tiles.
2. Downsample both VRTs to AROSICS_MAX_PX (avoids OOM on the nodata-mask read
   that AROSICS performs during init — shifts are in map units so downsampling
   the detection input does not affect shift accuracy).
3. Run AROSICS COREG_LOCAL.calculate_spatial_shifts() on the downsampled copies.
4. For each grid cell, interpolate the local shift from the AROSICS tie-point
   table using scipy linear interpolation (nearest-neighbour outside the hull).
   This honours the spatial variation in shift — one shift per tile, not one
   shift for the whole mosaic.
5. Apply per-cell shifts to full-resolution target tiles (zero pixel resampling).
6. Mosaic all corrected tiles into a Cloud-Optimized GeoTIFF.
"""

import os
import glob
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coregistration_utils as ict
from config import (
    REFERENCE_GLOB      as REFERENCE,
    TARGET_GLOB         as TARGET,
    BAND,
    N_CPUS,
    TILE_PX,
    AROSICS_GRID_RES,
    AROSICS_WIN_SIZE,
    AROSICS_MAX_SHIFT,
    AROSICS_MAX_PX,
    AROSICS_CORRECTION,
    AROSICS_SPLINE_EVAL_N,
)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coregistration_output_arosics")


def _write_translation_worker(args):
    tgt_vrt, cell, dx, dy, tiles_dir = args
    r, c, left, bottom, right, top, h_px, w_px, res = cell
    out = os.path.join(tiles_dir, f"tile_{r:04d}_{c:04d}.tif")
    ict.write_translation_tile(tgt_vrt, left, bottom, right, top,
                               h_px, w_px, dx, dy, res, out)
    return out


def _downsample(vrt_path, out_path, max_px):
    """Write a downsampled GeoTIFF capped at max_px on the longest edge."""
    from osgeo import gdal
    import rasterio
    with rasterio.open(vrt_path) as ds:
        w, h = ds.width, ds.height
    scale = min(1.0, max_px / max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    print(f"  {os.path.basename(vrt_path)}  {w}×{h}  →  {new_w}×{new_h} px")
    tmp = gdal.Translate(out_path, vrt_path,
                         width=new_w, height=new_h,
                         resampleAlg=gdal.GRA_Average,
                         format='GTiff',
                         creationOptions=['COMPRESS=DEFLATE'])
    tmp.FlushCache()
    tmp = None


def _raster_corners(path):
    """Return [[UL],[UR],[LR],[LL]] bounding-box corners from a raster's geotransform."""
    import rasterio
    with rasterio.open(path) as ds:
        b = ds.bounds
    return [[b.left, b.top], [b.right, b.top],
            [b.right, b.bottom], [b.left, b.bottom]]


def _detect_shifts(ref_small, tgt_small):
    """
    Run AROSICS on downsampled inputs and return a DataFrame of valid tie points
    (OUTLIER == False) after AROSICS' own quality filtering.
    """
    from arosics import COREG_LOCAL

    print(f"\nRunning AROSICS shift detection  "
          f"(grid_res={AROSICS_GRID_RES} px, "
          f"window={AROSICS_WIN_SIZE}, "
          f"max_shift={AROSICS_MAX_SHIFT} px, "
          f"band={BAND}) ...")

    CRL = COREG_LOCAL(
        ref_small, tgt_small,
        grid_res=AROSICS_GRID_RES,
        window_size=AROSICS_WIN_SIZE,
        max_shift=AROSICS_MAX_SHIFT,
        nodata=(0, 0),
        r_b4match=BAND,
        s_b4match=BAND,
        CPUs=N_CPUS,
        progress=True,
        data_corners_ref=_raster_corners(ref_small),
        data_corners_tgt=_raster_corners(tgt_small),
    )
    CRL.calculate_spatial_shifts()
    tbl = CRL.CoRegPoints_table

    # Use AROSICS' own filtering — mirrors Tie_Point_Grid.py line 430
    if "OUTLIER" in tbl.columns:
        valid_mask  = tbl["OUTLIER"].__eq__(False)
        n_valid     = int(valid_mask.sum())
        n_filtered  = int(tbl["OUTLIER"].__eq__(True).sum())
        n_unmatched = len(tbl) - n_valid - n_filtered
    else:
        valid_mask  = np.ones(len(tbl), bool)
        n_valid     = len(tbl)
        n_filtered  = n_unmatched = 0

    print(f"\n  AROSICS tie points : {len(tbl)}  "
          f"valid: {n_valid}  filtered: {n_filtered}  no-match: {n_unmatched}")

    if n_valid == 0:
        raise RuntimeError("AROSICS found no valid tie points — cannot determine shift.")

    return tbl[valid_mask].copy()


def _interpolate_cell_shifts(tbl_valid, grid):
    """
    Interpolate per-tile (dX, dY) from the AROSICS valid tie-point table.

    Uses scipy linear triangulation within the tie-point convex hull, with
    nearest-neighbour fallback for grid cells that fall outside it (typically
    the mosaic edges where tie points are sparse).

    Returns two arrays (pred_dx, pred_dy) aligned with `grid`.
    """
    from scipy.interpolate import griddata

    xs  = tbl_valid["X_MAP"].values.astype(float)
    ys  = tbl_valid["Y_MAP"].values.astype(float)
    dxs = tbl_valid["X_SHIFT_M"].values.astype(float)
    dys = tbl_valid["Y_SHIFT_M"].values.astype(float)

    # Centre of each grid cell in map coordinates
    cx = np.array([(cell[2] + cell[4]) / 2.0 for cell in grid])
    cy = np.array([(cell[3] + cell[5]) / 2.0 for cell in grid])
    query  = np.column_stack([cx, cy])
    points = np.column_stack([xs, ys])

    # Linear interpolation inside convex hull; zero shift outside.
    # Tiles outside the hull have no reference coverage — extrapolating shifts
    # pulls them away from the main mosaic, creating floating fragment artefacts.
    pred_dx = griddata(points, dxs, query, method='linear', fill_value=0.0)
    pred_dy = griddata(points, dys, query, method='linear', fill_value=0.0)

    outside = np.isnan(griddata(points, dxs, query, method='linear', fill_value=np.nan))
    if outside.any():
        print(f"  {outside.sum()} edge tile(s) outside tie-point coverage → zero shift")

    print(f"  Per-cell dX range : [{pred_dx.min():+.7f},  {pred_dx.max():+.7f}] map units")
    print(f"  Per-cell dY range : [{pred_dy.min():+.7f},  {pred_dy.max():+.7f}] map units")

    return pred_dx, pred_dy


def _apply_spline_correction(tbl_valid, tgt_vrt, tiles_dir):
    """
    Per-pixel thin-plate spline warp applied tile-by-tile.

    Fits one RBFInterpolator to all valid AROSICS tie points, then for every
    grid tile evaluates the spline at each pixel to get a per-pixel (dX, dY)
    displacement, and warps the tile with cv2.remap (bilinear interpolation).
    A small overlap buffer eliminates seam artefacts at tile boundaries.
    """
    import cv2
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
    xs  = tbl_valid["X_MAP"].values.astype(float)
    ys  = tbl_valid["Y_MAP"].values.astype(float)
    dxs = tbl_valid["X_SHIFT_M"].values.astype(float)
    dys = tbl_valid["Y_SHIFT_M"].values.astype(float)

    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import Delaunay

    pts  = np.column_stack([xs, ys])
    hull = Delaunay(pts)

    print("  Fitting thin-plate spline to tie points ...")
    spline_dx = RBFInterpolator(pts, dxs, kernel="thin_plate_spline")
    spline_dy = RBFInterpolator(pts, dys, kernel="thin_plate_spline")

    N       = AROSICS_SPLINE_EVAL_N   # coarse grid size per tile
    OVERLAP = 64

    corrected = []
    with rasterio.open(tgt_vrt) as src:
        gt_c    = src.transform.c
        gt_f    = src.transform.f
        px_x    = src.transform.a   # positive
        px_y    = src.transform.e   # negative
        n_bands = src.count
        dtype   = src.dtypes[0]
        nodata  = src.nodata if src.nodata is not None else 0
        crs     = src.crs
        full_w  = src.width
        full_h  = src.height

        grid = ict.compute_grid(tgt_vrt, TILE_PX)
        print(f"  Warping {len(grid)} tiles  "
              f"(spline eval {N}×{N} pts/tile → bilinear upsample) ...")

        for cell in tqdm(grid, desc="    Warping"):
            r, c, left, bottom, right, top, h_px, w_px, _ = cell
            out     = os.path.join(tiles_dir, f"tile_{r:04d}_{c:04d}.tif")
            tile_tf = from_bounds(left, bottom, right, top, w_px, h_px)
            tile_cx = (left + right)  / 2.0
            tile_cy = (top  + bottom) / 2.0

            # ── Outside hull: write tile unshifted ────────────────────────────
            if hull.find_simplex([[tile_cx, tile_cy]])[0] < 0:
                col_off = max(0, min(full_w - w_px,
                                     int(round((left - gt_c) / px_x))))
                row_off = max(0, min(full_h - h_px,
                                     int(round((top  - gt_f) / px_y))))
                data = src.read(window=Window(col_off, row_off, w_px, h_px))
                with rasterio.open(out, "w", driver="GTiff",
                                   height=h_px, width=w_px,
                                   count=n_bands, dtype=dtype,
                                   crs=crs, transform=tile_tf,
                                   compress="DEFLATE") as dst:
                    dst.write(data)
                corrected.append(out)
                continue

            # ── Evaluate spline on N×N coarse grid, upsample to full res ─────
            # GPS shift fields are smooth — N=16 (256 pts) is ~260,000× faster
            # than per-pixel evaluation with no meaningful loss of accuracy.
            c_x = np.linspace(left, right,  N)
            c_y = np.linspace(top,  bottom, N)
            cxx, cyy = np.meshgrid(c_x, c_y)
            q   = np.column_stack([cxx.ravel(), cyy.ravel()])
            dx  = cv2.resize(spline_dx(q).reshape(N, N).astype(np.float32),
                             (w_px, h_px), interpolation=cv2.INTER_LINEAR)
            dy  = cv2.resize(spline_dy(q).reshape(N, N).astype(np.float32),
                             (w_px, h_px), interpolation=cv2.INTER_LINEAR)

            # ── Buffered read to avoid black fringe at tile edges after warp ──
            col_off = max(0, int(round((left   - gt_c) / px_x)) - OVERLAP)
            row_off = max(0, int(round((top    - gt_f) / px_y)) - OVERLAP)
            col_end = min(full_w, int(round((right  - gt_c) / px_x)) + OVERLAP)
            row_end = min(full_h, int(round((bottom - gt_f) / px_y)) + OVERLAP)
            data    = src.read(window=Window(col_off, row_off,
                                            col_end - col_off,
                                            row_end - row_off))

            # ── Source pixel coordinates (vectorised, no meshgrid) ────────────
            buf_x0   = gt_c + col_off * px_x
            buf_y0   = gt_f + row_off * px_y
            base_col = (left - buf_x0) / px_x + 0.5
            base_row = (top  - buf_y0) / px_y + 0.5
            col_idx  = (base_col + np.arange(w_px)).astype(np.float32)
            row_idx  = (base_row + np.arange(h_px)).astype(np.float32)
            src_px_x = col_idx[np.newaxis, :] - (dx / px_x).astype(np.float32)
            src_px_y = row_idx[:, np.newaxis] - (dy / px_y).astype(np.float32)

            # ── Warp each band ────────────────────────────────────────────────
            warped = np.empty((n_bands, h_px, w_px), dtype=dtype)
            for b in range(n_bands):
                warped[b] = cv2.remap(
                    data[b].astype(np.float32), src_px_x, src_px_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=float(nodata),
                ).astype(dtype)

            with rasterio.open(out, "w", driver="GTiff",
                               height=h_px, width=w_px,
                               count=n_bands, dtype=dtype,
                               crs=crs, transform=tile_tf,
                               compress="DEFLATE") as dst:
                dst.write(warped)
            corrected.append(out)

    return corrected


def run(ref_glob, tgt_glob, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tiles_dir = os.path.join(output_dir, "corrected_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    ref_tiles = sorted(glob.glob(ref_glob))
    tgt_tiles = sorted(glob.glob(tgt_glob))
    if not ref_tiles:
        raise FileNotFoundError(f"No reference tiles matched: {ref_glob}")
    if not tgt_tiles:
        raise FileNotFoundError(f"No target tiles matched: {tgt_glob}")

    print(f"Reference : {len(ref_tiles)} tile(s)")
    print(f"Target    : {len(tgt_tiles)} tile(s)")

    ref_vrt = os.path.join(output_dir, "_reference_mosaic.vrt")
    tgt_vrt = os.path.join(output_dir, "_target_mosaic.vrt")
    ict.build_vrt(ref_tiles, ref_vrt)
    ict.build_vrt(tgt_tiles, tgt_vrt)

    # ── Step 1: downsample for AROSICS ────────────────────────────────────────
    print(f"\n[1/3] Downsampling inputs to ≤{AROSICS_MAX_PX} px for AROSICS ...")
    ref_small = os.path.join(output_dir, "_ref_arosics.tif")
    tgt_small = os.path.join(output_dir, "_tgt_arosics.tif")
    _downsample(ref_vrt, ref_small, AROSICS_MAX_PX)
    _downsample(tgt_vrt, tgt_small, AROSICS_MAX_PX)

    # ── Step 2: detect per-point shifts with AROSICS ──────────────────────────
    print(f"\n[2/3] AROSICS shift detection ...")
    tbl_valid = _detect_shifts(ref_small, tgt_small)

    for p in (ref_small, tgt_small):
        try:
            os.remove(p)
        except OSError:
            pass

    # ── Step 3: apply correction ───────────────────────────────────────────────
    if AROSICS_CORRECTION == "spline":
        print(f"\n[3/3] Spline warp correction (per-pixel thin-plate spline) ...")
        corrected = _apply_spline_correction(tbl_valid, tgt_vrt, tiles_dir)
    else:
        grid = ict.compute_grid(tgt_vrt, TILE_PX)
        print(f"\n[3/3] Translation correction — interpolating shifts for "
              f"{len(grid)} tiles ...")
        pred_dx, pred_dy = _interpolate_cell_shifts(tbl_valid, grid)
        work = [
            (tgt_vrt, cell, float(pred_dx[i]), float(pred_dy[i]), tiles_dir)
            for i, cell in enumerate(grid)
        ]
        corrected = []
        with ProcessPoolExecutor(max_workers=N_CPUS) as pool:
            futs = {pool.submit(_write_translation_worker, w): w for w in work}
            for f in tqdm(as_completed(futs), total=len(work), desc="    Writing"):
                corrected.append(f.result())

    # ── Step 4: mosaic → COG ──────────────────────────────────────────────────
    final = os.path.join(output_dir, "coregistered_final.tif")
    print(f"\n[4/4] Mosaicking {len(corrected)} tiles → {final}")
    ict.mosaic_cog(corrected, final, tmp_dir=output_dir)

    print(f"\nDone.  Final output: {final}")
    return final


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run(REFERENCE, TARGET, OUTPUT)
