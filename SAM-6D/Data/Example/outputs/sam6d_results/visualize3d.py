"""
3-D visualiser for a depth/RGB frame + pose JSON
– Assumes the depth PNG is a 16-bit image whose pixel values are in millimetres.
– Replace the camera-intrinsic numbers (fx, fy, cx, cy) with your real ones!
"""
import json
import os
import cv2
import numpy as np
import open3d as o3d
from pycocotools import mask as mask_utils     # RLE → binary mask

# ---------------------------------------------------------------------
# Files ----------------------------------------------------------------
img_path = os.environ.get("RGB_PATH")
if img_path is None:
  raise ValueError("RGB_PATH environment variable is not set.")
depth_path = os.environ.get("DEPTH_PATH")
if img_path is None:
  raise ValueError("RGB_PATH environment variable is not set.")


DEPTH_PATH = depth_path
RGB_PATH   = img_path
JSON_PATH  = "detection_pem.json"


# ---------------------------------------------------------------------
# Robust COCO-RLE decoder ---------------------------------------------
import numpy as np
from pycocotools import mask as mask_utils    # already imported earlier

# ---------------------------------------------------------------------
# Robust COCO-RLE decoder  (v2 – handles list or string)
# ---------------------------------------------------------------------
def coco_decode(rle):
    """
    Accepts either:
      • compressed RLE  –  {"counts": <bytes/str>, "size":[H,W]}
      • uncompressed RLE – {"counts": <list[int]>, "size":[H,W]}
    Returns an (H×W) uint8 binary mask.
    """
    if isinstance(rle["counts"], list):              # <- uncompressed
        # Re-implement COCO’s run-length decoding
        h, w = rle["size"]
        counts = rle["counts"]

        # Build 1-D mask then reshape column-major (‘Fortran’) to match COCO
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0                                     # 0 → background first
        for run in counts:
            if val == 1:
                flat[idx : idx + run] = 1
            idx += run
            val ^= 1                                # flip 0 ↔ 1 each run
        mask = flat.reshape((h, w), order="F")
    else:                                           # <- already compressed
        mask = mask_utils.decode(rle)[:, :, 0]      # (H,W,1) → (H,W)

    return mask



# ---------------------------------------------------------------------
# Camera intrinsics ----------------------------------------------------
#            fx  fy   cx   cy   (≈ RealSense D435 defaults – change!)
fx, fy, cx, cy = 615.0, 615.0, 320, 240
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                              fx=fx, fy=fy, cx=cx, cy=cy)

# ---------------------------------------------------------------------
# Load images ----------------------------------------------------------
depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED).astype(np.float32)  # mm
rgb_raw   = cv2.cvtColor(cv2.imread(RGB_PATH), cv2.COLOR_BGR2RGB)            # HxWx3

# Convert depth to metres and wrap in Open3D images
depth_m = depth_raw / 1_000.0
rgb_o3d   = o3d.geometry.Image(rgb_raw)
depth_o3d = o3d.geometry.Image(depth_m)

rgbd  = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
# Flip to conventional camera coords (+X right, +Y up, +Z forward)
pcd.transform([[1, 0, 0, 0],
               [0,-1, 0, 0],
               [0, 0,-1, 0],
               [0, 0, 0, 1]])

geoms = [pcd]                   # things to draw

# ---------------------------------------------------------------------
# Overlay each detection ----------------------------------------------
with open(JSON_PATH) as f:
    dets = json.load(f)

for det in dets:
    # --- pose ---------------------------------------------------------
    R = np.array(det["R"])                     # 3×3
    t = np.array(det["t"], dtype=float) / 1_000.0   # mm → m
    T = np.eye(4);  T[:3,:3] = R;  T[:3, 3] = t

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    frame.transform(T)
    geoms.append(frame)

    # --- object mask as its own coloured cloud -----------------------
    rle   = det["segmentation"]
    mask  = coco_decode(rle)               # <<< use robust helper here

    # If JSON resolution differs from depth, resize mask:
    if mask.shape[:2] != depth_raw.shape:
        mask = cv2.resize(mask.astype(np.uint8),
                          (depth_raw.shape[1], depth_raw.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    ys, xs = np.where(mask)
    zs = depth_m[ys, xs]
    xs = (xs - cx) * zs / fx
    ys = (ys - cy) * zs / fy
    obj_pts = np.column_stack((xs, ys, zs))

    obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    obj_pcd.paint_uniform_color([1, 0, 0])     # red highlight
    geoms.append(obj_pcd)

# ---------------------------------------------------------------------
# Show it --------------------------------------------------------------
print("Left-mouse: rotate | Scroll: zoom | Right-mouse: pan")
o3d.visualization.draw_geometries(geoms)
