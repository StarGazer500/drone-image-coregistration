"""
visualize_matches.py — Co-registration match / shift visualiser.

Set METHOD (line ~38) to select the visualisation mode:

  "homography" / "affine"
      Re-run SIFT + RANSAC on the same tiled grid used by run_homography.py /
      run_affine.py.  Produces per-cell side-by-side correspondence PNGs, a
      colour-coded match-map grid, and a GeoPackage with inlier/outlier lines.

  "arosics"
      Run AROSICS COREG_LOCAL.calculate_spatial_shifts() on the full VRT mosaic
      (no manual tiling).  Produces a tie-point shift-magnitude scatter map and
      a GeoPackage with tie_point_stats (Point) and shift_vectors (LineString).

Outputs are written to match_visualizations_<METHOD>/.
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
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coregistration_utils as ict

# ── Configuration ──────────────────────────────────────────────────────────────
from config import REFERENCE_GLOB as REFERENCE, TARGET_GLOB as TARGET
from config import TILE_PX, DETECT_PX, BAND, MAX_SHIFT_PX

METHOD     = "arosics"   # "homography", "affine", or "arosics"
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


# ── AROSICS visualisation ─────────────────────────────────────────────────────

def _downsample_for_arosics(vrt_path, out_path, max_px):
    """
    Write a downsampled GeoTIFF capped at max_px along the longest edge.
    AROSICS returns shifts in map units regardless of input resolution — the
    detected shift is the same whether the image is 4 096 px or 127 000 px.
    """
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


def _vrt_corners(path):
    """
    Return [[UL],[UR],[LR],[LL]] bounding-box corners from a raster's
    geotransform (no pixel reads).  Passed to AROSICS as data_corners_* to
    skip the automatic footprint computation.
    """
    import rasterio
    with rasterio.open(path) as ds:
        b = ds.bounds
    return [[b.left, b.top], [b.right, b.top],
            [b.right, b.bottom], [b.left, b.bottom]]


def _run_arosics_shifts(ref_vrt, tgt_vrt):
    """
    Detect shifts using AROSICS COREG_LOCAL.calculate_spatial_shifts().

    Both inputs are downsampled to AROSICS_MAX_PX before being passed to
    AROSICS so that the single-band nodata-mask read stays within RAM.
    Shifts are returned in map units, so the result is resolution-independent.
    Does NOT apply any correction.
    """
    from arosics import COREG_LOCAL
    from config import N_CPUS, AROSICS_GRID_RES, AROSICS_WIN_SIZE, AROSICS_MAX_SHIFT, AROSICS_MAX_PX

    ref_small = os.path.join(OUTPUT_DIR, "_ref_arosics.tif")
    tgt_small = os.path.join(OUTPUT_DIR, "_tgt_arosics.tif")

    print(f"Downsampling inputs to ≤{AROSICS_MAX_PX} px for AROSICS ...")
    _downsample_for_arosics(ref_vrt, ref_small, AROSICS_MAX_PX)
    _downsample_for_arosics(tgt_vrt, tgt_small, AROSICS_MAX_PX)

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
        data_corners_ref=_vrt_corners(ref_small),
        data_corners_tgt=_vrt_corners(tgt_small),
    )
    CRL.calculate_spatial_shifts()

    for p in (ref_small, tgt_small):
        try:
            os.remove(p)
        except OSError:
            pass

    return CRL.CoRegPoints_table


def _save_arosics_shift_map(table, out_path):
    """
    Scatter map of AROSICS tie points coloured by absolute shift magnitude.

    AROSICS stores three distinct values in the OUTLIER column:
      False (-0)   : valid tie point
      True  (1)    : matched but filtered out by quality checks
      -9999        : outFillVal — no match found (never had a valid shift)
    Using .__eq__(False) mirrors the check AROSICS uses internally.
    """
    xs = table["X_MAP"].values.astype(float)
    ys = table["Y_MAP"].values.astype(float)

    valid      = table["OUTLIER"].__eq__(False).values   # 119 valid
    filtered   = table["OUTLIER"].__eq__(True).values    # quality-filtered
    unmatched  = ~valid & ~filtered                      # no-match (-9999)

    shifts = (table.loc[table["OUTLIER"].__eq__(False), "ABS_SHIFT"]
              .values.astype(float)
              if "ABS_SHIFT" in table.columns else np.array([]))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")

    if valid.any() and len(shifts) > 0:
        vmax = max(float(shifts.max()), 1e-9)
        sc = ax.scatter(xs[valid], ys[valid], c=shifts,
                        cmap="RdYlGn_r", s=40, zorder=3,
                        vmin=0, vmax=vmax, label=f"Valid ({valid.sum()})")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("ABS_SHIFT (map units)", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    if filtered.any():
        ax.scatter(xs[filtered], ys[filtered], c="#ff6633",
                   marker="x", s=40, linewidths=1.2, zorder=4,
                   label=f"Filtered outlier ({filtered.sum()})")

    if unmatched.any():
        ax.scatter(xs[unmatched], ys[unmatched], c="#555555",
                   marker=".", s=15, zorder=2,
                   label=f"No match ({unmatched.sum()})")

    n_valid     = int(valid.sum())
    n_filtered  = int(filtered.sum())
    n_unmatched = int(unmatched.sum())
    med_shift   = float(np.median(shifts)) if len(shifts) > 0 else 0.0

    ax.text(0.02, 0.98,
            f"Total: {len(table)}  |  valid: {n_valid}  "
            f"filtered: {n_filtered}  no-match: {n_unmatched}  |  "
            f"median shift: {med_shift:.3e} map units",
            transform=ax.transAxes, va="top", fontsize=8, color="white",
            bbox=dict(fc="#333333", alpha=0.75, boxstyle="round,pad=0.3"))

    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_xlabel("X (map units)", color="white")
    ax.set_ylabel("Y (map units)", color="white")
    ax.set_title("AROSICS Local Co-registration — Tie-point Shift Map",
                 fontsize=11, color="white")
    ax.tick_params(colors="white", labelsize=7)
    legend = ax.legend(fontsize=8, facecolor="#333333", edgecolor="#555555")
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close(fig)
    print(f"  Shift map saved → {out_path}")


def _save_arosics_geopackage(table, ref_vrt, out_path):
    """
    Write two GeoPackage layers from the AROSICS CoRegPoints table:

    tie_point_stats   Point      One point per tie point with shift attributes.
    shift_vectors     LineString Start = tie point position; end = shifted position.
                                 Metre shifts are converted to CRS units so the
                                 lines are geographically correct in QGIS.
    """
    from osgeo import gdal as _gdal, ogr as _ogr, osr as _osr

    ds = _gdal.Open(ref_vrt, _gdal.GA_ReadOnly)
    srs = _osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    ds = None
    is_geographic = bool(srs.IsGeographic())

    if os.path.exists(out_path):
        os.remove(out_path)

    drv  = _ogr.GetDriverByName("GPKG")
    gpkg = drv.CreateDataSource(out_path)

    attr_fields = [
        ("X_SHIFT_M",   _ogr.OFTReal),
        ("Y_SHIFT_M",   _ogr.OFTReal),
        ("ABS_SHIFT",   _ogr.OFTReal),
        ("ANGLE",       _ogr.OFTReal),
        ("RELIABILITY", _ogr.OFTReal),
        ("OUTLIER",     _ogr.OFTInteger),
    ]

    lyr_pts = gpkg.CreateLayer("tie_point_stats", srs=srs, geom_type=_ogr.wkbPoint)
    for fname, ftype in [("X_MAP", _ogr.OFTReal), ("Y_MAP", _ogr.OFTReal)] + attr_fields:
        lyr_pts.CreateField(_ogr.FieldDefn(fname, ftype))

    lyr_vec = gpkg.CreateLayer("shift_vectors", srs=srs, geom_type=_ogr.wkbLineString)
    for fname, ftype in attr_fields:
        lyr_vec.CreateField(_ogr.FieldDefn(fname, ftype))

    def _col(row, name, default=0.0):
        return float(row[name]) if name in table.columns and row[name] == row[name] else default

    for _, row in table.iterrows():
        x     = float(row["X_MAP"])
        y     = float(row["Y_MAP"])
        dx_m  = _col(row, "X_SHIFT_M")
        dy_m  = _col(row, "Y_SHIFT_M")
        abs_s = _col(row, "ABS_SHIFT")
        angle = _col(row, "ANGLE")
        rel   = _col(row, "RELIABILITY")
        # Preserve the raw OUTLIER value: 0 (valid), 1 (filtered), -9999 (unmatched)
        try:
            out = int(row["OUTLIER"]) if "OUTLIER" in table.columns else 0
        except (ValueError, TypeError):
            out = 0

        # Point — write for every tie point
        pt = _ogr.Geometry(_ogr.wkbPoint)
        pt.AddPoint_2D(x, y)
        feat = _ogr.Feature(lyr_pts.GetLayerDefn())
        feat.SetGeometry(pt)
        feat["X_MAP"] = x;      feat["Y_MAP"]     = y
        feat["X_SHIFT_M"] = dx_m; feat["Y_SHIFT_M"] = dy_m
        feat["ABS_SHIFT"]   = abs_s; feat["ANGLE"]       = angle
        feat["RELIABILITY"] = rel;   feat["OUTLIER"]     = out
        lyr_pts.CreateFeature(feat)

        # Shift vector — only for matched points (skip outFillVal=-9999 rows which
        # have no valid shift and would produce ~10 km lines in QGIS for WGS84 data)
        if abs(dx_m) >= 9999:
            continue

        if is_geographic:
            lat_rad = np.radians(y)
            dx_crs = dx_m / (111320.0 * max(np.cos(lat_rad), 1e-9))
            dy_crs = dy_m / 111320.0
        else:
            dx_crs, dy_crs = dx_m, dy_m

        line = _ogr.Geometry(_ogr.wkbLineString)
        line.AddPoint_2D(x, y)
        line.AddPoint_2D(x + dx_crs, y + dy_crs)
        feat = _ogr.Feature(lyr_vec.GetLayerDefn())
        feat.SetGeometry(line)
        feat["X_SHIFT_M"] = dx_m; feat["Y_SHIFT_M"] = dy_m
        feat["ABS_SHIFT"]   = abs_s; feat["ANGLE"]       = angle
        feat["RELIABILITY"] = rel;   feat["OUTLIER"]     = out
        lyr_vec.CreateFeature(feat)

    gpkg.FlushCache()
    gpkg = None
    print(f"  GeoPackage saved → {out_path}")
    print(f"    Layers: tie_point_stats, shift_vectors")


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

    if METHOD == "arosics":
        # ── AROSICS path: no manual tiling, AROSICS places its own grid ──────
        table = _run_arosics_shifts(ref_vrt, tgt_vrt)

        n_total = len(table)
        if "OUTLIER" in table.columns:
            valid_mask  = table["OUTLIER"].__eq__(False)
            n_valid     = int(valid_mask.sum())
            n_filtered  = int(table["OUTLIER"].__eq__(True).sum())
            n_unmatched = n_total - n_valid - n_filtered
            med_shift   = (float(table.loc[valid_mask, "ABS_SHIFT"].median())
                           if "ABS_SHIFT" in table.columns else 0.0)
        else:
            n_valid = n_total; n_filtered = 0; n_unmatched = 0; med_shift = 0.0
        print(f"\n{n_valid}/{n_total} tie points valid  "
              f"(filtered: {n_filtered}  no-match: {n_unmatched})  "
              f"median shift: {med_shift:.3e} map units")

        map_path = os.path.join(OUTPUT_DIR, "match_map.png")
        _save_arosics_shift_map(table, map_path)

        if SAVE_GEOPACKAGE:
            gpkg_path = os.path.join(OUTPUT_DIR, "matches_arosics.gpkg")
            _save_arosics_geopackage(table, ref_vrt, gpkg_path)

        # Cleanup temp VRTs
        for p in (ref_vrt, tgt_vrt):
            try:
                os.remove(p)
            except OSError:
                pass

        print(f"\nDone.  Output folder: {OUTPUT_DIR}/")
        print(f"  Shift map  : match_map.png")
        if SAVE_GEOPACKAGE:
            print(f"  GeoPackage : matches_arosics.gpkg")
            print(f"    Layers   : tie_point_stats, shift_vectors")

    else:
        # ── SIFT path: tiled grid detection ──────────────────────────────────
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
