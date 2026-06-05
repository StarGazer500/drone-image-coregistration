"""
visualize_matches.py — SIFT correspondence visualiser.

For every grid cell that passes detection, produces a side-by-side PNG showing:
  • Left panel  : reference crop (grayscale)
  • Right panel : target crop (histogram-matched, grayscale)
  • Grey lines  : good matches that RANSAC rejected
  • Green lines : RANSAC inlier correspondences
  • Title bar   : cell position, detected shift, inlier count

Also produces match_map.png — a colour-coded grid overview:
  • Dark green  : high inliers (≥ 100)
  • Light green : low inliers (10 – 99)
  • Red         : detection failed (pixels present but too few features/inliers)
  • Dark grey   : no valid pixels in one or both mosaics

Outputs are written to OUTPUT_DIR/.
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coregistration_utils as ict

# ── Configuration ──────────────────────────────────────────────────────────────
from config import REFERENCE_GLOB as REFERENCE, TARGET_GLOB as TARGET
from config import TILE_PX, DETECT_PX, BAND, MAX_SHIFT_PX

METHOD     = "affine"   # "homography" or "affine" — controls the RANSAC step
OUTPUT_DIR = f"match_visualizations_{METHOD}"

# How many inlier lines to draw per cell (None = all)
MAX_LINES  = 80

# Set True to save a PNG for every matched cell; False for the map only
SAVE_CELL_IMAGES = True

# Set True to export match lines and cell stats to a GeoPackage for QGIS
SAVE_GEOPACKAGE  = True


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_cell(ref_vrt, tgt_vrt, left, bottom, right, top,
                 h_px, w_px, band, detect_px, method="homography", max_shift_px=None):
    """
    Run SIFT + FLANN + findHomography for one cell.

    Returns a dict with keys:
        status      : "matched" | "failed" | "nodata"
        ref_u8      : uint8 reference crop (H×W)
        tgt_u8      : uint8 target crop, histogram-matched (H×W)
        kp_r, kp_t  : keypoint lists
        good        : good DMatch list (passed ratio test)
        inliers     : bool array over good (True = RANSAC inlier)
        dx_map, dy_map : shift in map units
        n_inliers   : int
        message     : human-readable status string
        extent      : (left, bottom, right, top) geographic extent of the cell
        det_size    : (det_w, det_h) detection-resolution pixel dimensions
    """
    scale = min(1.0, detect_px / max(h_px, w_px))
    det_h = max(1, int(round(h_px * scale)))
    det_w = max(1, int(round(w_px * scale)))

    ref_arr = ict.read_tile_band(ref_vrt, left, bottom, right, top, det_h, det_w, band)
    tgt_arr = ict.read_tile_band(tgt_vrt, left, bottom, right, top, det_h, det_w, band)
    ref_u8  = ict.to_uint8(ref_arr)
    tgt_u8  = ict.to_uint8(tgt_arr)

    ext  = (left, bottom, right, top)
    dsz  = (det_w, det_h)

    def _ret(status, msg, kp_r=None, kp_t=None, good=None,
             inliers=None, dx=0.0, dy=0.0, n=0, tgt=None):
        return {"status": status, "message": msg,
                "ref_u8": ref_u8, "tgt_u8": tgt if tgt is not None else tgt_u8,
                "kp_r": kp_r or [], "kp_t": kp_t or [],
                "good": good or [], "inliers": inliers if inliers is not None else np.array([]),
                "dx_map": dx, "dy_map": dy, "n_inliers": n,
                "extent": ext, "det_size": dsz}

    if ref_u8.max() == 0 or tgt_u8.max() == 0:
        return _ret("nodata", "no valid pixels")

    tgt_matched = ict.histogram_match(ref_u8, tgt_u8)

    sift        = cv2.SIFT_create(nfeatures=5000)
    kp_r, des_r = sift.detectAndCompute(ref_u8,      None)
    kp_t, des_t = sift.detectAndCompute(tgt_matched, None)

    if des_r is None or des_t is None or len(kp_r) < 4 or len(kp_t) < 4:
        return _ret("failed",
                    f"too few keypoints (ref={len(kp_r or [])}, tgt={len(kp_t or [])})",
                    kp_r=kp_r, kp_t=kp_t, tgt=tgt_matched)

    flann   = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    matches = flann.knnMatch(des_r, des_t, k=2)
    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return _ret("failed", f"only {len(good)} good matches",
                    kp_r=kp_r, kp_t=kp_t, good=good,
                    inliers=np.zeros(len(good), bool), tgt=tgt_matched)

    src_pts = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good])

    if method == "affine":
        _, mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    else:
        _, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return _ret("failed", "RANSAC returned no mask",
                    kp_r=kp_r, kp_t=kp_t, good=good,
                    inliers=np.zeros(len(good), bool), tgt=tgt_matched)

    inliers = mask.ravel().astype(bool)
    if max_shift_px is not None:
        mags    = np.sqrt(((src_pts - dst_pts) ** 2).sum(axis=1))
        inliers = inliers & (mags <= max_shift_px)
    n_in    = int(inliers.sum())

    if n_in < 10:
        return _ret("failed", f"only {n_in} inliers after displacement filter",
                    kp_r=kp_r, kp_t=kp_t, good=good,
                    inliers=inliers, n=n_in, tgt=tgt_matched)

    dx_det = float(np.median(src_pts[inliers, 0] - dst_pts[inliers, 0]))
    dy_det = float(np.median(src_pts[inliers, 1] - dst_pts[inliers, 1]))
    dx_map =  dx_det * (right - left) / det_w
    dy_map = -dy_det * (top   - bottom) / det_h

    return _ret("matched",
                f"{n_in} inliers  dX={dx_map:+.7f}  dY={dy_map:+.7f}",
                kp_r=kp_r, kp_t=kp_t, good=good, inliers=inliers,
                dx=dx_map, dy=dy_map, n=n_in, tgt=tgt_matched)


def _save_cell_image(result, row, col, out_path, max_lines=None):
    """
    Draw SIFT correspondences on a side-by-side image and save as PNG.
    Grey lines = RANSAC outliers; green lines = inliers.
    """
    ref_u8  = result["ref_u8"]
    tgt_u8  = result["tgt_u8"]
    kp_r    = result["kp_r"]
    kp_t    = result["kp_t"]
    good    = result["good"]
    inliers = result["inliers"]
    status  = result["status"]

    # Separate inlier / outlier match lists
    in_matches  = [m for m, keep in zip(good, inliers) if     keep]
    out_matches = [m for m, keep in zip(good, inliers) if not keep]

    if max_lines is not None:
        in_matches  = in_matches[:max_lines]
        out_matches = out_matches[:max(0, max_lines - len(in_matches))]

    h, w = ref_u8.shape[:2]
    canvas = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)

    # Convert grayscale to BGR for colour drawing
    ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
    tgt_bgr = cv2.cvtColor(tgt_u8, cv2.COLOR_GRAY2BGR)
    canvas[:, :w]         = ref_bgr
    canvas[:, w + 4:]     = tgt_bgr
    canvas[:, w:w + 4]    = 40   # thin divider

    def draw_matches_on_canvas(matches, color):
        for m in matches:
            pt_r = (int(kp_r[m.queryIdx].pt[0]),
                    int(kp_r[m.queryIdx].pt[1]))
            pt_t = (int(kp_t[m.trainIdx].pt[0]) + w + 4,
                    int(kp_t[m.trainIdx].pt[1]))
            cv2.line(canvas, pt_r, pt_t, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_r, 3, color, -1)
            cv2.circle(canvas, pt_t, 3, color, -1)

    draw_matches_on_canvas(out_matches, (100, 100, 100))   # grey: outliers
    draw_matches_on_canvas(in_matches,  (0,   220,  60))   # green: inliers

    # Header bar
    bar_h = 36
    header = np.full((bar_h, canvas.shape[1], 3), 30, dtype=np.uint8)
    if status == "matched":
        label = (f"[{row:03d},{col:03d}]  {result['message']}"
                 f"  |  grey={len(out_matches)} outliers  green={len(in_matches)} inliers")
        color = (80, 220, 80)
    else:
        label = f"[{row:03d},{col:03d}]  {status.upper()} — {result['message']}"
        color = (80, 80, 220)

    cv2.putText(header, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)

    final = np.vstack([header, canvas])

    # Column labels
    for x, lbl in ((w // 2, "REFERENCE"), (w + 4 + w // 2, "TARGET (hist-matched)")):
        cv2.putText(final, lbl, (x - 60, bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, final)


def _save_match_map(grid, cell_results, out_path):
    """
    Colour-coded grid: dark green = many inliers, red = failed, dark grey = nodata.
    """
    rows = sorted(set(r for r, c, *_ in grid))
    cols = sorted(set(c for r, c, *_ in grid))
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    # Build arrays for coloring
    STATUS = np.zeros((n_rows, n_cols), dtype=int)   # 0=nodata,1=failed,2=matched
    INLIERS = np.zeros((n_rows, n_cols), dtype=int)

    for r, c, *_ in grid:
        res = cell_results.get((r, c))
        if res is None:
            STATUS[r, c] = 0
        elif res["status"] == "nodata":
            STATUS[r, c] = 0
        elif res["status"] == "failed":
            STATUS[r, c] = 1
            INLIERS[r, c] = res["n_inliers"]
        else:
            STATUS[r, c] = 2
            INLIERS[r, c] = res["n_inliers"]

    cell_w = max(2.0, 12.0 / n_cols)
    cell_h = max(2.0, 10.0 / n_rows)
    fig, ax = plt.subplots(figsize=(n_cols * cell_w, n_rows * cell_h))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_title("SIFT Match Map  —  grid cells coloured by inlier count", fontsize=11)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="white", linewidth=0.4)

    max_in = max(INLIERS.max(), 1)
    for r in range(n_rows):
        for c in range(n_cols):
            s = STATUS[r, c]
            n = INLIERS[r, c]
            if s == 0:
                fc = "#2a2a2a"
            elif s == 1:
                fc = "#8b1a1a" if n == 0 else "#c04040"
            else:
                t = min(1.0, n / max(max_in, 100))
                r_val = 0.1 + 0.4 * (1 - t)
                g_val = 0.55 + 0.45 * t
                b_val = 0.1
                fc = (r_val, g_val, b_val)

            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=fc, edgecolor="white", linewidth=0.3)
            ax.add_patch(rect)

            if s >= 1 and n > 0:
                ax.text(c, r, str(n), ha="center", va="center",
                        fontsize=max(4, min(8, int(cell_w * 5))),
                        color="white", fontweight="bold")
            elif s == 1 and n == 0:
                ax.text(c, r, "✗", ha="center", va="center",
                        fontsize=max(4, min(8, int(cell_w * 5))),
                        color="#ff8080")

    patches = [
        mpatches.Patch(color="#1e7a1e", label="Matched (high inliers)"),
        mpatches.Patch(color="#5ab55a", label="Matched (low inliers)"),
        mpatches.Patch(color="#c04040", label="Failed (pixels present)"),
        mpatches.Patch(color="#2a2a2a", label="No valid pixels"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="grey")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Match map saved → {out_path}")


# ── GeoPackage export ─────────────────────────────────────────────────────────

def _save_geopackage(grid, cell_results, ref_vrt, out_path, method):
    """
    Write three layers to a GeoPackage:

    match_lines_inliers   LineString  ref-keypoint → tgt-keypoint for every
                                      RANSAC inlier.  Load in QGIS over the
                                      reference image to see where features moved.
    match_lines_outliers  LineString  Same for RANSAC outliers (grey in PNG).
    cell_stats            Polygon     One rectangle per grid cell with
                                      n_inliers, status, dx_map, dy_map as attrs.
    """
    from osgeo import gdal as _gdal, ogr as _ogr, osr as _osr

    # Read CRS from the reference VRT
    ds = _gdal.Open(ref_vrt, _gdal.GA_ReadOnly)
    srs = _osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    ds = None

    if os.path.exists(out_path):
        os.remove(out_path)

    drv  = _ogr.GetDriverByName("GPKG")
    gpkg = drv.CreateDataSource(out_path)

    def _make_layer(name, geom_type, fields):
        lyr = gpkg.CreateLayer(name, srs=srs, geom_type=geom_type)
        for fname, ftype in fields:
            lyr.CreateField(_ogr.FieldDefn(fname, ftype))
        return lyr

    line_fields = [
        ("row",       _ogr.OFTInteger),
        ("col",       _ogr.OFTInteger),
        ("n_inliers", _ogr.OFTInteger),
        ("dx_map",    _ogr.OFTReal),
        ("dy_map",    _ogr.OFTReal),
        ("dist_px",   _ogr.OFTReal),
        ("method",    _ogr.OFTString),
    ]
    lyr_in  = _make_layer("match_lines_inliers",  _ogr.wkbLineString, line_fields)
    lyr_out = _make_layer("match_lines_outliers", _ogr.wkbLineString, line_fields)

    cell_fields = [
        ("row",       _ogr.OFTInteger),
        ("col",       _ogr.OFTInteger),
        ("status",    _ogr.OFTString),
        ("n_inliers", _ogr.OFTInteger),
        ("dx_map",    _ogr.OFTReal),
        ("dy_map",    _ogr.OFTReal),
        ("method",    _ogr.OFTString),
    ]
    lyr_cell = _make_layer("cell_stats", _ogr.wkbPolygon, cell_fields)

    def _px_to_geo(px, py, left, top, right, bottom, dw, dh):
        """Convert detection-pixel (px,py) to geographic coordinates."""
        x = left + px * (right - left) / dw
        y = top  - py * (top - bottom) / dh
        return x, y

    for row, col, left, bottom, right, top, *_ in grid:
        res_d = cell_results.get((row, col))
        if res_d is None:
            continue

        extent   = res_d.get("extent", (left, bottom, right, top))
        det_size = res_d.get("det_size", (1, 1))
        dw, dh   = det_size
        L, B, R, T = extent

        # ── Cell boundary polygon ──────────────────────────────────────────
        ring = _ogr.Geometry(_ogr.wkbLinearRing)
        ring.AddPoint(L, T); ring.AddPoint(R, T)
        ring.AddPoint(R, B); ring.AddPoint(L, B)
        ring.AddPoint(L, T)
        poly = _ogr.Geometry(_ogr.wkbPolygon)
        poly.AddGeometry(ring)

        feat = _ogr.Feature(lyr_cell.GetLayerDefn())
        feat.SetGeometry(poly)
        feat["row"]       = row
        feat["col"]       = col
        feat["status"]    = res_d["status"]
        feat["n_inliers"] = res_d["n_inliers"]
        feat["dx_map"]    = res_d["dx_map"]
        feat["dy_map"]    = res_d["dy_map"]
        feat["method"]    = method
        lyr_cell.CreateFeature(feat)

        # ── Match lines ────────────────────────────────────────────────────
        kp_r    = res_d.get("kp_r", [])
        kp_t    = res_d.get("kp_t", [])
        good    = res_d.get("good", [])
        inliers = res_d.get("inliers", np.array([]))

        if len(good) == 0 or len(inliers) == 0:
            continue

        for m, is_inlier in zip(good, inliers):
            px_r, py_r = kp_r[m.queryIdx].pt
            px_t, py_t = kp_t[m.trainIdx].pt

            x_r, y_r = _px_to_geo(px_r, py_r, L, T, R, B, dw, dh)
            x_t, y_t = _px_to_geo(px_t, py_t, L, T, R, B, dw, dh)

            line = _ogr.Geometry(_ogr.wkbLineString)
            line.AddPoint(x_r, y_r)
            line.AddPoint(x_t, y_t)

            dist_px = float(np.sqrt((px_r - px_t)**2 + (py_r - py_t)**2))

            lyr = lyr_in if is_inlier else lyr_out
            feat = _ogr.Feature(lyr.GetLayerDefn())
            feat.SetGeometry(line)
            feat["row"]       = row
            feat["col"]       = col
            feat["n_inliers"] = res_d["n_inliers"]
            feat["dx_map"]    = res_d["dx_map"]
            feat["dy_map"]    = res_d["dy_map"]
            feat["dist_px"]   = dist_px
            feat["method"]    = method
            lyr.CreateFeature(feat)

    gpkg.FlushCache()
    gpkg = None
    print(f"  GeoPackage saved → {out_path}")
    print(f"    Layers: match_lines_inliers, match_lines_outliers, cell_stats")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_tiles = sorted(glob.glob(REFERENCE))
    tgt_tiles = sorted(glob.glob(TARGET))
    if not ref_tiles:
        raise FileNotFoundError(f"No reference tiles matched: {REFERENCE}")
    if not tgt_tiles:
        raise FileNotFoundError(f"No target tiles matched: {TARGET}")

    print(f"Reference : {len(ref_tiles)} tiles")
    print(f"Target    : {len(tgt_tiles)} tiles")

    ref_vrt = os.path.join(OUTPUT_DIR, "_ref.vrt")
    tgt_vrt = os.path.join(OUTPUT_DIR, "_tgt.vrt")
    ict.build_vrt(ref_tiles, ref_vrt)
    ict.build_vrt(tgt_tiles, tgt_vrt)

    grid    = ict.compute_grid(tgt_vrt, TILE_PX)
    n_cells = len(grid)
    print(f"Grid      : {n_cells} cells\n")

    cell_results = {}
    n_matched = 0

    for i, (row, col, left, bottom, right, top, h_px, w_px, res) in enumerate(grid):
        print(f"  [{row:03d},{col:03d}]  ({i+1}/{n_cells})", end="  ")
        result = _detect_cell(ref_vrt, tgt_vrt, left, bottom, right, top,
                               h_px, w_px, BAND, DETECT_PX,
                               method=METHOD, max_shift_px=MAX_SHIFT_PX)
        cell_results[(row, col)] = result
        print(result["message"])

        if result["status"] == "matched":
            n_matched += 1
            if SAVE_CELL_IMAGES:
                img_path = os.path.join(OUTPUT_DIR,
                                         f"match_{row:03d}_{col:03d}.png")
                _save_cell_image(result, row, col, img_path, max_lines=MAX_LINES)

    print(f"\n{n_matched}/{n_cells} cells matched.")

    map_path = os.path.join(OUTPUT_DIR, "match_map.png")
    _save_match_map(grid, cell_results, map_path)

    if SAVE_GEOPACKAGE:
        gpkg_path = os.path.join(OUTPUT_DIR, f"matches_{METHOD}.gpkg")
        _save_geopackage(grid, cell_results, ref_vrt, gpkg_path, METHOD)

    # Cleanup temp VRTs
    for p in (ref_vrt, tgt_vrt):
        try:
            os.remove(p)
        except OSError:
            pass

    print(f"\nDone.  Output folder: {OUTPUT_DIR}/")
    if SAVE_CELL_IMAGES:
        print(f"  Per-cell images : match_<row>_<col>.png  ({n_matched} files)")
    print(f"  Grid overview   : match_map.png")
    if SAVE_GEOPACKAGE:
        print(f"  GeoPackage      : matches_{METHOD}.gpkg")


if __name__ == "__main__":
    main()
