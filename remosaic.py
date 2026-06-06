"""
Re-run only the mosaic step on already-written corrected tiles.
Use after mosaic_cog was fixed to support rotated GeoTransforms.

Edit OUTPUT_DIR to match whichever run you want to re-mosaic.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coregistration_utils as ict

OUTPUT_DIR = "coregistration_output_affine_selective"  # change to whichever run you want to re-mosaic

tiles_dir = os.path.join(OUTPUT_DIR, "corrected_tiles")
tile_paths = sorted(glob.glob(os.path.join(tiles_dir, "tile_*.tif")))
if not tile_paths:
    raise FileNotFoundError(f"No tiles found in {tiles_dir}")

print(f"Found {len(tile_paths)} tiles in {tiles_dir}")

# Remove stale output and temp files from the previous failed run
for fname in ("coregistered_final.tif", "coregistered_final.tif.ovr.tmp",
              "_final_mosaic.vrt"):
    p = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed stale file: {p}")

final = os.path.join(OUTPUT_DIR, "coregistered_final.tif")
print(f"\nMosaicking {len(tile_paths)} tiles → {final}")
ict.mosaic_cog(tile_paths, final, tmp_dir=OUTPUT_DIR)
print(f"\nDone.  Final output: {final}")
