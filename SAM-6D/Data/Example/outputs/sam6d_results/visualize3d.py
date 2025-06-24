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

from pycocotools import mask as mask_utils
import numpy as np

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

# for det in dets:
#     # --- pose ---------------------------------------------------------
#     R = np.array(det["R"])                     # 3×3
#     t = np.array(det["t"], dtype=float) / 1_000.0   # mm → m
#     T = np.eye(4);  T[:3,:3] = R;  T[:3, 3] = t

#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
#     frame.transform(T)
#     geoms.append(frame)

#     # --- object mask as its own coloured cloud -----------------------
#     rle   = det["segmentation"]
#     mask  = coco_decode(rle)
#     # If JSON resolution differs from depth, resize mask:
#     if mask.shape[:2] != depth_raw.shape:
#         mask = cv2.resize(mask.astype(np.uint8),
#                           (depth_raw.shape[1], depth_raw.shape[0]),
#                           interpolation=cv2.INTER_NEAREST).astype(bool)

#     ys, xs = np.where(mask)
#     zs = depth_m[ys, xs]
#     xs = (xs - cx) * zs / fx
#     ys = (ys - cy) * zs / fy
#     obj_pts = np.column_stack((xs, ys, zs))

#     obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
#     obj_pcd.paint_uniform_color([1, 0, 0])     # red highlight
#     geoms.append(obj_pcd)

# # ---------------------------------------------------------------------
# # Show it --------------------------------------------------------------
# print("Left-mouse: rotate | Scroll: zoom | Right-mouse: pan")
# o3d.visualization.draw_geometries(geoms)

#     rle   = det["segmentation"]
#     mask  = coco_decode(rle)

#     if mask.shape[:2] != depth_raw.shape:           # keep shapes aligned
#         mask = cv2.resize(mask.astype(np.uint8),
#                           (depth_raw.shape[1], depth_raw.shape[0]),
#                           interpolation=cv2.INTER_NEAREST).astype(bool)

#     ys, xs = np.where(mask)
#     zs = depth_m[ys, xs]
#     xs3d = (xs - cx) * zs / fx
#     ys3d = (ys - cy) * zs / fy
#     obj_pts = np.column_stack((xs3d, ys3d, zs))

#     obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))

#     # ★ NEW – colour each mask point from the original RGB image ★
#     rgb_vals = rgb_raw[ys, xs] / 255.0              # 0-1 float
#     obj_pcd.colors = o3d.utility.Vector3dVector(rgb_vals)

#     geoms.append(obj_pcd)

# # ---------------------------------------------------------------------
# # Show it --------------------------------------------------------------
# print("Left-mouse: rotate  |  Scroll: zoom  |  Right-mouse: pan")
# o3d.visualization.draw_geometries(geoms)

# ---------------------------------------------------------------------
# Load images ----------------------------------------------------------
depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)  # 16-bit → uint16
depth_raw = depth_raw.astype(np.float32)                  # mm → float32
rgb_raw   = cv2.cvtColor(cv2.imread(RGB_PATH), cv2.COLOR_BGR2RGB)

depth_scale = 1000.0                                      # mm → metres
valid_depth = depth_raw > 0                               # ### NEW ###

# Convert depth to metres *only where valid*
depth_m = np.where(valid_depth, depth_raw / depth_scale, np.nan)

# Wrap in Open3D -------------------------------------------------------
rgb_o3d   = o3d.geometry.Image(rgb_raw)
depth_o3d = o3d.geometry.Image(depth_raw)                 # keep raw mm here

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
           rgb_o3d, depth_o3d,
           depth_scale=depth_scale,  # tells Open3D the units
           depth_trunc=5.0,          # clip anything farther than 5 m
           convert_rgb_to_intensity=False)

FLIP_YZ = np.array([[1, 0, 0, 0],        # X  stays  +
                    [0,-1, 0, 0],        # Y  down  →  Y up
                    [0, 0,-1, 0],        # Z  forward → Z back (Open3D default)
                    [0, 0, 0, 1]], dtype=float)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
# pcd.transform([[1, 0, 0, 0],
#                [0,-1, 0, 0],
#                [0, 0,-1, 0],
#                [0, 0, 0, 1]])

# geoms = [pcd]

# ---------------------------------------------------------------------
# After creating the scene point cloud, add the CAMERA pose -----------
pcd.transform(FLIP_YZ)

geoms = [pcd]

# ★ NEW – camera frame (size 0.2 m) ★
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20)
cam_frame.transform(FLIP_YZ)
geoms.append(cam_frame)




# ---------------------------------------------------------------------
# Per-object overlay ---------------------------------------------------
for det in dets:
    # pose frame (unchanged) …
    # ---------------------------------------------------- mask points
    rle   = det["segmentation"]
    mask  = coco_decode(rle).astype(bool)

    if mask.shape[:2] != depth_raw.shape:
        mask = cv2.resize(mask.astype(np.uint8),
                          (depth_raw.shape[1], depth_raw.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    mask &= valid_depth                # ### NEW – drop zero-depth pixels ###

    ys, xs = np.where(mask)
    if xs.size == 0:                   # nothing left → skip
        continue

    z = depth_raw[ys, xs] / depth_scale            # metres
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    obj_pts = np.column_stack((x, y, z))

    # bounding box (or oriented box) ----------------------------------
    # obj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    # aabb   = obj_pc.get_axis_aligned_bounding_box()
    # aabb.color = (1, 0, 0)
    # geoms.append(aabb)

    # # optional: draw the coloured object cloud
    # obj_pc.colors = o3d.utility.Vector3dVector(rgb_raw[ys, xs] / 255.0)
    # geoms.append(obj_pc)

    # ---------------------------------------------------------------------
    # Inside the per-detection loop – REPLACE the pose-frame + bbox block
    # ---------------------------------------------------------------------
    # 1) Build 3-D points of the mask  (unchanged up to `obj_pts`)

    # obj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))

    # # 2) Size-aware pose frame -------------------------------------------
    # #    • use bbox diagonal to pick an easy-to-see axis length
    # diag_len = np.linalg.norm(obj_pc.get_max_bound() - obj_pc.get_min_bound())
    # axis_len = max(0.03, 0.15 * diag_len)     # min 3 cm, else 15 % of bbox

    # pose_T = np.eye(4)
    # pose_T[:3, :3] = np.array(det["R"])
    # pose_T[:3,  3] = np.array(det["t"], dtype=float) / 1_000.0   # mm → m
    # pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)
    # pose_frame.transform(pose_T @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
    # geoms.append(pose_frame)


    obj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    obj_pc.transform(FLIP_YZ)       # ★ NEW – flip object cloud

    # re-hook colours
    obj_pc.colors = o3d.utility.Vector3dVector(rgb_raw[ys, xs] / 255.0)

    # build bbox *after* flipping so it encloses the shown points
    aabb = obj_pc.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    geoms.extend([obj_pc, aabb])

    # pose frame – flip **after** applying R|t
    pose_T = np.eye(4)
    pose_T[:3,:3] = np.array(det["R"])
    pose_T[:3, 3]  = np.array(det["t"], dtype=float) / 1_000.0
    pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=max(0.03, 0.15 * np.linalg.norm(aabb.get_extent())))
    pose_frame.transform(FLIP_YZ @ pose_T)   # flip last
    geoms.append(pose_frame)

    
    # 3) Bounding box -----------------------------------------------------
    # Choose ONE of the two lines below:

    aabb = obj_pc.get_axis_aligned_bounding_box()      # axis-aligned        ← default
    # obb  = obj_pc.get_oriented_bounding_box()        # oriented to points  ← uncomment for OBB

    bbox = aabb                                        # or obb
    bbox.color = (1, 0, 0)
    geoms.append(bbox)

    # 4) Optional – keep the coloured object cloud
    obj_pc.colors = o3d.utility.Vector3dVector(rgb_raw[ys, xs] / 255.0)
    geoms.append(obj_pc)


# ---------------------------------------------------------------------
o3d.visualization.draw_geometries(geoms)

