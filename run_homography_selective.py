"""
main_homography_tiled_selective.py — Per-tile SIFT translation co-registration,
correction applied ONLY to cells with successful shift detection.

Difference from main_homography_tiled.py
-----------------------------------------
Cells that fail detection (no valid pixels, too few keypoints, too few RANSAC
inliers) are written with ZERO shift — left exactly as they are in the target
mosaic.  No median fallback is applied.

This avoids incorrectly shifting areas that already align well or that contain
no reliable features for matching.  Use this when you trust that unmatched cells
are genuinely well-aligned (or are no-data regions where the shift is irrelevant).

Algorithm
---------
1. Build VRTs from all reference and target tiles.
2. Partition the target mosaic into a uniform grid (TILE_PX × TILE_PX cells).
3. For every cell, read BOTH mosaics at the same bounds & pixel dimensions.
4. Detect translation shift via SIFT + findHomography (8 DOF) + RANSAC.
   Translation extracted as median of inlier point differences.
5. Detected cells → their own per-cell shift.
   Failed cells   → zero shift (tile written at original position).
6. Mosaic all cells into a single Cloud-Optimized GeoTIFF.
"""

import os
import sys
import glob
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coregistration_utils as ict

# ── Configuration ─────────────────────────────────────────────────────────────
from config import REFERENCE_GLOB as REFERENCE, TARGET_GLOB as TARGET
from config import TILE_PX, DETECT_PX, BAND, N_CPUS, MAD_K, MAX_SHIFT_PX
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coregistration_output_homography_selective")


# ── Per-cell write worker ─────────────────────────────────────────────────────

def _write_worker(args):
    tgt_vrt, cell, dx, dy, tiles_dir = args
    r, c, left, bottom, right, top, h_px, w_px, res = cell
    out = os.path.join(tiles_dir, f"tile_{r:04d}_{c:04d}.tif")
    ict.write_translation_tile(tgt_vrt, left, bottom, right, top,
                               h_px, w_px, dx, dy, res, out)
    return out


# ── Per-cell shift detection (subprocess worker) ──────────────────────────────

def _detect(args):
    """
    Detect translation shift for one grid cell via SIFT + findHomography.

    findHomography (8 DOF) RANSAC classifies inliers; translation is extracted
    as the median of inlier point differences (robust to remaining outliers).

    Returns (row, col, dx_map, dy_map, info_str)
         or (row, col, None,   None,   error_str).
    """
    row, col, left, bottom, right, top, h_px, w_px, res, ref_vrt, tgt_vrt, band, detect_px, max_shift_px = args
    try:
        import cv2
        import numpy as _np
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import coregistration_utils as _ict

        scale  = min(1.0, detect_px / max(h_px, w_px))
        det_h  = max(1, int(round(h_px * scale)))
        det_w  = max(1, int(round(w_px * scale)))

        ref_arr = _ict.read_tile_band(ref_vrt, left, bottom, right, top, det_h, det_w, band)
        tgt_arr = _ict.read_tile_band(tgt_vrt, left, bottom, right, top, det_h, det_w, band)
        ref_u8  = _ict.to_uint8(ref_arr)
        tgt_u8  = _ict.to_uint8(tgt_arr)

        if ref_u8.max() == 0 or tgt_u8.max() == 0:
            raise RuntimeError("cell has no valid pixels in one or both mosaics")

        tgt_matched = _ict.histogram_match(ref_u8, tgt_u8)

        sift         = cv2.SIFT_create(nfeatures=5000)
        kp_r, des_r  = sift.detectAndCompute(ref_u8,      None)
        kp_t, des_t  = sift.detectAndCompute(tgt_matched, None)
        if des_r is None or des_t is None or len(kp_r) < 4 or len(kp_t) < 4:
            raise RuntimeError(
                f"too few keypoints (ref={len(kp_r) if kp_r else 0}, "
                f"tgt={len(kp_t) if kp_t else 0})"
            )

        flann   = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        matches = flann.knnMatch(des_r, des_t, k=2)
        good    = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            raise RuntimeError(f"only {len(good)} good matches after ratio test")

        src_pts = _np.float32([kp_r[m.queryIdx].pt for m in good])
        dst_pts = _np.float32([kp_t[m.trainIdx].pt for m in good])
        _, mask  = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        inliers  = mask.ravel().astype(bool)
        mags     = _np.sqrt(((src_pts - dst_pts) ** 2).sum(axis=1))
        inliers  = inliers & (mags <= max_shift_px)
        if inliers.sum() < 10:
            raise RuntimeError(f"only {inliers.sum()} RANSAC inliers (need ≥10 for reliable shift)")

        dx_det = float(_np.median(src_pts[inliers, 0] - dst_pts[inliers, 0]))
        dy_det = float(_np.median(src_pts[inliers, 1] - dst_pts[inliers, 1]))
        dx_map =  dx_det * (right - left) / det_w
        dy_map = -dy_det * (top   - bottom) / det_h
        return row, col, dx_map, dy_map, f"{inliers.sum()} inliers"

    except Exception as exc:
        return row, col, None, None, str(exc)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(ref_glob, tgt_glob, output_dir, tile_px=8192, band=1, n_cpus=4):
    os.makedirs(output_dir, exist_ok=True)
    tiles_dir = os.path.join(output_dir, "corrected_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    ref_tiles = sorted(glob.glob(ref_glob))
    tgt_tiles = sorted(glob.glob(tgt_glob))
    if not ref_tiles:
        raise FileNotFoundError(f"No reference tiles matched: {ref_glob}")
    if not tgt_tiles:
        raise FileNotFoundError(f"No target tiles matched: {tgt_glob}")

    print(f"Reference : {len(ref_tiles)} tiles")
    print(f"Target    : {len(tgt_tiles)} tiles")

    ref_vrt = os.path.join(output_dir, "_reference_mosaic.vrt")
    tgt_vrt = os.path.join(output_dir, "_target_mosaic.vrt")
    ict.build_vrt(ref_tiles, ref_vrt)
    ict.build_vrt(tgt_tiles, tgt_vrt)

    grid    = ict.compute_grid(tgt_vrt, tile_px)
    n_cells = len(grid)
    print(f"Grid      : {n_cells} cells  ({tile_px}×{tile_px} px each)\n")

    # ── Step 1: detect per-cell shifts in parallel ───────────────────────────
    print(f"[1/3] Detecting shifts  ({n_cpus} workers)...")
    det_args = [(*cell, ref_vrt, tgt_vrt, band, DETECT_PX, MAX_SHIFT_PX) for cell in grid]
    shifts   = {}

    with ProcessPoolExecutor(max_workers=n_cpus) as pool:
        futs = {pool.submit(_detect, a): (a[0], a[1]) for a in det_args}
        for f in tqdm(as_completed(futs), total=n_cells, desc="    Cells"):
            row, col, dx, dy, info = f.result()
            key = (row, col)
            if dx is not None:
                shifts[key] = (dx, dy)
                tqdm.write(f"    [{row:03d},{col:03d}]  dX={dx:+.7f}  dY={dy:+.7f}  ({info})")
            else:
                shifts[key] = None
                tqdm.write(f"    [{row:03d},{col:03d}]  FAILED — {info}")

    valid = [v for v in shifts.values() if v is not None]
    if len(valid) >= 3:
        dxs    = np.array([v[0] for v in valid])
        dys    = np.array([v[1] for v in valid])
        med_x0 = float(np.median(dxs))
        med_y0 = float(np.median(dys))
        mad_x  = float(np.median(np.abs(dxs - med_x0)))
        mad_y  = float(np.median(np.abs(dys - med_y0)))
        thr_x  = MAD_K * mad_x if mad_x > 0 else np.inf
        thr_y  = MAD_K * mad_y if mad_y > 0 else np.inf
        keep   = ((np.abs(dxs - med_x0) <= thr_x) &
                  (np.abs(dys - med_y0) <= thr_y))
        n_out  = int((~keep).sum())
        med_x  = float(np.median(dxs[keep])) if keep.any() else med_x0
        med_y  = float(np.median(dys[keep])) if keep.any() else med_y0
        n_ok   = int(keep.sum())
    else:
        med_x = float(np.median([v[0] for v in valid])) if valid else 0.0
        med_y = float(np.median([v[1] for v in valid])) if valid else 0.0
        n_ok  = len(valid)
        n_out = 0

    n_detected = len(valid)
    print(f"\n    {n_detected}/{n_cells} cells detected  |  "
          f"consensus shift dX={med_x:+.7f}  dY={med_y:+.7f}  (from {n_ok} inlier(s))")
    if n_out:
        print(f"    {n_out} outlier cell(s) excluded from consensus by MAD filter (k={MAD_K}).")
    print(f"    {n_detected} cell(s) will be individually corrected.")
    print(f"    {n_cells - n_detected} cell(s) left uncorrected (zero shift — no detection).")

    # ── Step 2: write corrected tiles ────────────────────────────────────────
    print(f"\n[2/3] Writing {n_cells} tiles...")
    work = [
        (tgt_vrt, cell,
         shifts[(cell[0], cell[1])][0] if shifts[(cell[0], cell[1])] is not None else 0.0,
         shifts[(cell[0], cell[1])][1] if shifts[(cell[0], cell[1])] is not None else 0.0,
         tiles_dir)
        for cell in grid
    ]
    corrected = []
    with ProcessPoolExecutor(max_workers=N_CPUS) as pool:
        futs = {pool.submit(_write_worker, w): w for w in work}
        for f in tqdm(as_completed(futs), total=len(work), desc="    Writing"):
            corrected.append(f.result())

    # ── Step 3: mosaic → single COG ──────────────────────────────────────────
    final = os.path.join(output_dir, "coregistered_final.tif")
    print(f"\n[3/3] Mosaicking {len(corrected)} tiles → {final}")
    ict.mosaic_cog(corrected, final, tmp_dir=output_dir)

    print(f"\nDone.  Final output: {final}")
    return final


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run(
        ref_glob   = REFERENCE,
        tgt_glob   = TARGET,
        output_dir = OUTPUT,
        tile_px    = TILE_PX,
        band       = BAND,
        n_cpus     = N_CPUS,
    )
