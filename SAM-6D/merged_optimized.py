#!/usr/bin/env python
import os
import sys
import logging
import json
from functools import lru_cache

import numpy as np
import torch
import trimesh
from PIL import Image
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
import pycocotools.mask as cocomask
from torchvision import transforms
from types import SimpleNamespace
from omegaconf import OmegaConf

import time

# ─── MULTI-OBJECT CONFIGURATION ────────────────────────────────────────────────
# Configuration for multiple object types
MULTI_OBJECT_CONFIG = {
    "landingpole": {
        "cad_path": "Data/Example/mesh/obj_landingpole.ply",
        "template_dir": "Data/Example/outputs/templates",
        "category_id": 1,
    },
    # Add more object types here:
    # "drone": {
    #     "cad_path": "Data/Example/mesh/obj_drone.ply", 
    #     "template_dir": "Data/Example/outputs/templates_drone",
    #     "category_id": 2,
    # },
    # "building": {
    #     "cad_path": "Data/Example/mesh/obj_building.ply",
    #     "template_dir": "Data/Example/outputs/templates_building", 
    #     "category_id": 3,
    # }
}

# Enhanced filtering - process multiple object types
PROCESS_ALL_OBJECTS = False  # Set to True to process all detected objects
MAX_OBJECTS_PER_TYPE = 3     # Maximum objects per category to process

# ─── VISUALIZATION CONFIGURATION ────────────────────────────────────────────────
# Toggle visualization features (set to False to disable specific features)
SHOW_2D_BBOX = True          # Show 2D bounding boxes
SHOW_3D_BBOX = True          # Show 3D bounding boxes with pose
SHOW_POSE_AXES = True        # Show coordinate axes at object center
SHOW_SEGMENTATION = True     # Show segmentation masks
SHOW_POSE_INFO = True        # Show pose information text (rotation angles)
SHOW_SCORE_INFO = True       # Show detection scores
SHOW_DISTANCE_INFO = True    # Show distance information
SHOW_OBJECT_ID = True        # Show object IDs

# Advanced visualization settings (inspired by visualize.py)
VISUALIZATION_CONFIG = {
    'show_3d_bbox': True,          # Show 3D bounding boxes
    'show_pose_axes': True,        # Show coordinate axes at object center
    'show_pose_info': True,        # Show pose information text
    'show_distance_info': True,    # Show distance information
    'show_score_info': True,       # Show detection scores
    'show_object_id': True,        # Show object IDs
    'text_color': (255, 255, 0),   # Yellow for text
    'bbox_colors': [               # Colors for different detections
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ],
    'axis_thickness': 4,           # Thickness of pose axes
    'bbox_thickness': 3,           # Thickness of 3D bbox lines
    'text_scale': 0.6,             # Scale of text
}

# Visualization parameters
MASK_ALPHA = 0.3             # Transparency for masks (0.0 = transparent, 1.0 = opaque)
TEXT_COLOR = (255, 255, 0)   # Yellow for text (B, G, R format)
BBOX_2D_COLOR = (255, 255, 0) # Yellow for 2D boxes
POSE_AXIS_LENGTH = 0.05      # Length of pose axes (relative to object size)

# ─── USER CONFIGURATION ─────────────────────────────────────────────────────────
OUTPUT_DIR         = "output/"
TEMPLATE_DIR       = "Data/Example/outputs/templates"  
CAD_PATH           = "Data/Example/mesh/obj_landingpole.ply"
RGB_PATH           = "Data/Example/rgb/rgb_landingpole_mid.png"
DEPTH_PATH         = "Data/Example/depth/depth_landingpole_mid.png"
CAM_PATH           = "Data/Example/camera_intrinsics/camera.json"

FASTSAM_CHECKPT    = os.path.join(
    os.path.dirname(__file__),
    "Instance_Segmentation_Model","checkpoints","FastSAM","FastSAM-x.pt"
)
FASTSAM_CFG        = {"iou_threshold":0.30, "conf_threshold":0.50, "max_det":50}
FASTSAM_WIDTH      = 1024

PEM_CHECKPT        = os.path.join(
    os.path.dirname(__file__),
    "Pose_Estimation_Model","checkpoints","sam-6d-pem-base.pth"
)
DET_SCORE_THRESH   = 0.2
TOP_K              = 5

RGB_MASK_FLAG           = False

# ─── PERFORMANCE OPTIMIZATION GLOBALS ───────────────────────────────────────────
# Global caches to avoid reloading models and data
_fastsam_model = None
_pem_model = None
_config_cache = None
_template_cache = None
_static_data_cache = None

# ─── BOILERPLATE ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PEM_CONFIG_PATH = os.path.join(BASE_DIR, "Pose_Estimation_Model", "config","base.yaml")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Instance_Segmentation_Model"))
pem_root = os.path.join(BASE_DIR, "Pose_Estimation_Model")
sys.path.insert(0, os.path.join(pem_root, "provider"))
sys.path.insert(0, os.path.join(pem_root, "utils"))
sys.path.insert(0, os.path.join(pem_root, "model"))
sys.path.insert(0, os.path.join(pem_root, "model", "pointnet2"))

# ─── IMPORTS ────────────────────────────────────────────────────────────────────
from segment_anything.utils.amg import rle_to_mask
from Instance_Segmentation_Model.model.fast_sam import FastSAM
from Instance_Segmentation_Model.utils.poses.pose_utils import (
    get_obj_poses_from_template_level,
    load_index_level_in_level2,
)
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23

# **Always these two**:
from Pose_Estimation_Model.utils.data_utils import (
    load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose,
)
from Pose_Estimation_Model.utils.draw_utils import draw_detections
from Pose_Estimation_Model.model.pose_estimation_model import Net as PEMNet

inv_rgb_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229, 1/0.224, 1/0.225],
    ),
])
rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ─── OPTIMIZATION: MODEL INITIALIZATION CACHE ──────────────────────────────────
def get_fastsam_model():
    """Initialize FastSAM model once and cache it"""
    global _fastsam_model
    if _fastsam_model is None:
        logging.info("→ Initializing FastSAM (first time)")
        cfg = SimpleNamespace(**FASTSAM_CFG)
        _fastsam_model = FastSAM(
            checkpoint_path      = FASTSAM_CHECKPT,
            config               = cfg,
            segmentor_width_size = FASTSAM_WIDTH,
            device               = device,
        )
    return _fastsam_model

def get_pem_model():
    """Initialize PEM model once and cache it"""
    global _pem_model, _config_cache
    if _pem_model is None:
        logging.info("→ Initializing PEM model (first time)")
        _config_cache = OmegaConf.load(PEM_CONFIG_PATH)
        _pem_model = PEMNet(_config_cache.model).to(device).eval()
        ckpt = torch.load(PEM_CHECKPT, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        _pem_model.load_state_dict(state_dict, strict=False)
    return _pem_model, _config_cache

# ─── OPTIMIZATION: STATIC DATA CACHE ────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_static_data():
    """Cache static data that doesn't change between runs"""
    # Load camera parameters
    cam = json.load(open(CAM_PATH))
    K = np.array(cam["cam_K"]).reshape(3,3)
    
    # Load and preprocess images
    rgb_pil = Image.open(RGB_PATH).convert("RGB")
    rgb_np = np.array(rgb_pil)
    
    img = load_im(RGB_PATH)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=2)
    img = img.astype(np.uint8)
    
    depth = load_im(DEPTH_PATH).astype(np.float32) * cam["depth_scale"] / 1000.0
    pts3 = get_point_cloud_from_depth(depth, K)
    
    return {
        'cam': cam,
        'K': K,
        'rgb_pil': rgb_pil,
        'rgb_np': rgb_np,
        'img': img,
        'depth': depth,
        'pts3': pts3
    }

@lru_cache(maxsize=1)
def get_model_points(cfg):
    """Cache model points sampling"""
    mesh = trimesh.load_mesh(CAD_PATH)
    model_pts = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    rad = np.max(np.linalg.norm(model_pts, axis=1))
    return model_pts, rad

# ─── OPTIMIZATION: TEMPLATE CACHE ───────────────────────────────────────────────
def get_template_cached(path, cfg, tem_index=1):
    """Cached version of get_template to avoid repeated file I/O"""
    cache_key = (path, tem_index, cfg.img_size, cfg.n_sample_template_point, cfg.rgb_mask_flag)
    
    global _template_cache
    if _template_cache is None:
        _template_cache = {}
    
    if cache_key not in _template_cache:
        rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
        choose = (mask_cropped>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= cfg.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
        choose_final = choose[choose_idx]
        xyz_final = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose_final, :]

        rgb_choose = get_resize_rgb_choose(choose_final, [y1, y2, x1, x2], cfg.img_size)
        
        _template_cache[cache_key] = (rgb_transformed, rgb_choose, xyz_final)
    
    return _template_cache[cache_key]

def get_templates_cached(path, cfg):
    """Cached version of get_templates with pre-computed tensor conversion"""
    cache_key = (path, cfg.n_template_view, cfg.img_size, cfg.n_sample_template_point, cfg.rgb_mask_flag)
    
    global _template_cache
    if _template_cache is None:
        _template_cache = {}
    
    templates_key = f"templates_{cache_key}"
    if templates_key not in _template_cache:
        total_views = 42
        all_tem, all_tem_pts, all_tem_choose = [], [], []

        for v in range(cfg.n_template_view):
            idx = int(total_views / cfg.n_template_view * v)
            tem, tem_choose, tem_pts = get_template_cached(path, cfg, idx)
            # Pre-convert to tensors and move to device
            all_tem.append(torch.FloatTensor(tem).unsqueeze(0).to(device))
            all_tem_choose.append(torch.LongTensor(tem_choose).unsqueeze(0).to(device))
            all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).to(device))
        
        _template_cache[templates_key] = (all_tem, all_tem_pts, all_tem_choose)
    
    return _template_cache[templates_key]

# ─── ADVANCED VISUALIZATION HELPERS (Based on visualize.py) ─────────────────────

def decode_rle_mask(rle, height, width):
    """Decode RLE mask to binary mask"""
    counts = rle['counts']
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    
    # Parse counts string to list of integers
    if isinstance(counts, str):
        counts = list(map(int, counts.split()))
    
    size = rle['size']
    
    # Initialize empty mask
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    
    # Decode RLE
    idx = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Fill regions (odd indices)
            mask[idx:idx+count] = 1
        idx += count
    
    return mask.reshape(size[0], size[1])

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (in degrees)"""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.degrees([x, y, z])

def create_3d_bbox_from_2d_center(bbox_2d, depth_est, K, aspect_ratio=1.0):
    """Create 3D bounding box corners based on 2D bbox center and estimated depth"""
    x, y, w, h = bbox_2d
    
    # Calculate 2D bounding box center
    center_2d_x = x + w / 2
    center_2d_y = y + h / 2
    
    # Back-project 2D center to 3D space at given depth
    K_inv = np.linalg.inv(K)
    center_2d_homogeneous = np.array([center_2d_x, center_2d_y, 1.0])
    center_3d = depth_est * (K_inv @ center_2d_homogeneous)
    
    # Estimate object dimensions based on 2D bbox size and depth
    obj_width = w * depth_est / K[0, 0]
    obj_height = h * depth_est / K[1, 1]
    obj_depth = min(obj_width, obj_height) * aspect_ratio
    
    # Define 3D box corners
    half_w, half_h, half_d = obj_width/2, obj_height/2, obj_depth/2
    
    box_3D_local = np.array([
        [-half_w, -half_h, -half_d],  # bottom face
        [ half_w, -half_h, -half_d],
        [ half_w,  half_h, -half_d],
        [-half_w,  half_h, -half_d],
        [-half_w, -half_h,  half_d],  # top face
        [ half_w, -half_h,  half_d],
        [ half_w,  half_h,  half_d],
        [-half_w,  half_h,  half_d],
    ]).T
    
    # Translate box to 3D center position
    box_3D = box_3D_local + center_3d.reshape(3, 1)
    
    return box_3D, center_3d

def visualize_seg(rgb_pil, dets, out_file):
    """Enhanced segmentation visualization"""
    img = np.array(rgb_pil)
    img_with_masks = img.copy()
    
    for i, det in enumerate(dets):
        if 'segmentation' in det and 'counts' in det['segmentation']:
            seg = det['segmentation']
            try:
                mask = decode_rle_mask(seg, seg['size'][0], seg['size'][1])
                
                # Create colored mask overlay
                color = distinctipy.get_colors(len(dets))[i % len(distinctipy.get_colors(len(dets)))]
                color_rgb = [int(255*c) for c in color]
                
                # Apply mask with transparency
                for c in range(3):
                    img_with_masks[:,:,c] = np.where(
                        mask,
                        MASK_ALPHA * color_rgb[c] + (1-MASK_ALPHA) * img_with_masks[:,:,c],
                        img_with_masks[:,:,c]
                    )
                
                # Add edges
                edge = binary_dilation(canny(mask), np.ones((2,2)))
                img_with_masks[edge] = [255, 255, 255]
                
            except Exception as e:
                print(f"Warning: Could not decode mask for detection {i}: {e}")
    
    # Save visualization
    vis_img = Image.fromarray(np.uint8(img_with_masks))
    out = Image.new("RGB", (img.shape[1] + vis_img.width, img.shape[0]))
    out.paste(Image.fromarray(img), (0, 0))
    out.paste(vis_img, (img.shape[1], 0))
    out.save(out_file)
    return out

def visualize_pose_advanced(rgb_np, detections, K, out_file):
    """Advanced pose visualization based on visualize.py logic"""
    img = rgb_np.copy()
    img_with_masks = rgb_np.copy()
    
    for i, obj in enumerate(detections):
        # Extract detection data
        bbox_2d = obj.get('bbox', [0, 0, 100, 100])
        score = obj.get('score', 0.0)
        R = np.array(obj['R']) if 'R' in obj else np.eye(3)
        t = np.array(obj['t']).reshape(3, 1) if 't' in obj else np.zeros((3, 1))
        segmentation = obj.get('segmentation', {})
        
        x, y, w, h = bbox_2d
        
        # Decode and overlay segmentation mask
        if SHOW_SEGMENTATION and 'counts' in segmentation and 'size' in segmentation:
            try:
                mask = decode_rle_mask(segmentation, segmentation['size'][0], segmentation['size'][1])
                
                # Create colored mask overlay
                color = np.array([np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)])
                mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_colored[mask == 1] = color
                
                # Blend mask with image
                img_with_masks = cv2.addWeighted(img_with_masks, 1-MASK_ALPHA, mask_colored, MASK_ALPHA, 0)
            except:
                pass
        
        # Draw 2D bounding box
        if SHOW_2D_BBOX:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), BBOX_2D_COLOR, 2)
            cv2.rectangle(img_with_masks, (int(x), int(y)), (int(x + w), int(y + h)), BBOX_2D_COLOR, 2)
        
        # Calculate pose information
        euler_angles = rotation_matrix_to_euler(R)
        distance = np.linalg.norm(t)
        
        # Add text information
        text_y_offset = int(y - 10)
        
        if SHOW_SCORE_INFO:
            cv2.putText(img, f'Score: {score:.3f}', (int(x), text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            cv2.putText(img_with_masks, f'Score: {score:.3f}', (int(x), text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            text_y_offset -= 15
        
        if SHOW_DISTANCE_INFO:
            cv2.putText(img, f'Dist: {distance:.0f}mm', (int(x), text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            cv2.putText(img_with_masks, f'Dist: {distance:.0f}mm', (int(x), text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            text_y_offset -= 15
        
        if SHOW_POSE_INFO:
            cv2.putText(img, f'Rot: ({euler_angles[0]:.1f}°,{euler_angles[1]:.1f}°,{euler_angles[2]:.1f}°)', 
                        (int(x), text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            cv2.putText(img_with_masks, f'Rot: ({euler_angles[0]:.1f}°,{euler_angles[1]:.1f}°,{euler_angles[2]:.1f}°)', 
                        (int(x), text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            text_y_offset -= 15
        
        if SHOW_OBJECT_ID:
            cv2.putText(img, f'ID: {i}', (int(x), int(y + h + 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            cv2.putText(img_with_masks, f'ID: {i}', (int(x), int(y + h + 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
        # Draw 3D bounding box
        if SHOW_3D_BBOX:
            depth_estimate = np.linalg.norm(t)
            box_3D, center_3d = create_3d_bbox_from_2d_center(bbox_2d, depth_estimate, K, aspect_ratio=0.8)
            
            # Transform 3D box to camera coordinates
            box_cam = R @ box_3D + t
            proj = K @ box_cam
            
            # Check if points are in front of camera
            valid_depth = proj[2] > 0
            if np.all(valid_depth):
                proj = proj[:2] / proj[2]
                proj = proj.T.astype(int)
                
                # Draw 3D box with different colors
                color_offset = i * 50 % 255
                box_color = (255 - color_offset, color_offset, 128)
                edge_thickness = 3
                
                for target_img in [img, img_with_masks]:
                    # Draw bottom face
                    bottom_color = tuple(int(c * 0.7) for c in box_color)
                    for j in range(4):
                        cv2.line(target_img, tuple(proj[j]), tuple(proj[(j + 1) % 4]), bottom_color, edge_thickness)
                    
                    # Draw top face
                    top_color = box_color
                    for j in range(4, 8):
                        cv2.line(target_img, tuple(proj[j]), tuple(proj[4 + (j + 1) % 4]), top_color, edge_thickness)
                    
                    # Draw vertical edges
                    vertical_color = tuple(int(c * 0.85) for c in box_color)
                    for j in range(4):
                        cv2.line(target_img, tuple(proj[j]), tuple(proj[j + 4]), vertical_color, edge_thickness)
                    
                    # Draw center point
                    center_2d = (int(x + w / 2), int(y + h / 2))
                    cv2.circle(target_img, center_2d, 4, (255, 255, 255), -1)
                    cv2.circle(target_img, center_2d, 5, box_color, 2)
        
        # Draw coordinate axes
        if SHOW_POSE_AXES:
            # Calculate axis lengths based on bounding box
            depth_estimate = np.linalg.norm(t)
            obj_width = w * depth_estimate / K[0, 0] / 1000.0
            obj_height = h * depth_estimate / K[1, 1] / 1000.0
            obj_depth = min(obj_width, obj_height) * 0.8
            
            x_axis_length = obj_width / 2
            y_axis_length = obj_height / 2
            z_axis_length = obj_depth / 2
            
            axes = np.float32([
                [0, 0, 0],                      # Origin
                [x_axis_length, 0, 0],          # X axis
                [0, y_axis_length, 0],          # Y axis
                [0, 0, z_axis_length]           # Z axis
            ]).T
            
            # Transform axes to camera coordinates
            pts = R @ axes + t
            pts_proj = K @ pts
            
            # Check if axis points are in front of camera
            if np.all(pts_proj[2] > 0):
                pts_proj = (pts_proj[:2] / pts_proj[2]).T.astype(int)
                
                origin = tuple(pts_proj[0])
                x_end = tuple(pts_proj[1])
                y_end = tuple(pts_proj[2])
                z_end = tuple(pts_proj[3])
                
                # Draw axes on both images
                for target_img in [img, img_with_masks]:
                    # X axis - Red
                    cv2.arrowedLine(target_img, origin, x_end, (0, 0, 255), 4, tipLength=0.2)
                    cv2.putText(target_img, 'X', (x_end[0] + 5, x_end[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Y axis - Green
                    cv2.arrowedLine(target_img, origin, y_end, (0, 255, 0), 4, tipLength=0.2)
                    cv2.putText(target_img, 'Y', (y_end[0] + 5, y_end[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Z axis - Blue
                    cv2.arrowedLine(target_img, origin, z_end, (255, 0, 0), 4, tipLength=0.2)
                    cv2.putText(target_img, 'Z', (z_end[0] + 5, z_end[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Create side-by-side visualization
    ppil = Image.fromarray(np.uint8(img))
    rpil = Image.fromarray(np.uint8(img_with_masks))
    
    out = Image.new("RGB", (img.shape[1] + ppil.width, max(img.shape[0], ppil.height)))
    out.paste(rpil, (0, 0))
    out.paste(ppil, (img.shape[1], 0))
    out.save(out_file)
    return out

# Legacy function for backward compatibility
def visualize_pose(rgb_np, R, t, model_pts, K, out_file):
    """Legacy pose visualization - redirects to advanced version"""
    # Create detection object in expected format
    detection = {
        'bbox': [0, 0, rgb_np.shape[1]//4, rgb_np.shape[0]//4],  # Default bbox
        'score': 1.0,
        'R': R.tolist() if hasattr(R, 'tolist') else R,
        't': t.flatten().tolist() if hasattr(t, 'flatten') else t,
        'segmentation': {}
    }
    return visualize_pose_advanced(rgb_np, [detection], K, out_file)

# ─── OPTIMIZED SEGMENTATION (ISM) ───────────────────────────────────────────────
def run_segmentation_optimized():
    sam = get_fastsam_model()
    static_data = get_static_data()
    
    # Use cached RGB data
    rgb_pil = static_data['rgb_pil']
    rgb_np = static_data['rgb_np']

    # **PASS array** to get back a dict of masks+boxes
    mask_data = sam.generate_masks(rgb_np)
    dets = Detections(mask_data)

    # Move to CPU for numpy ops (batch operation)
    with torch.no_grad():
        dets.boxes = dets.boxes.cpu()
        dets.masks = dets.masks.cpu()

    # Optimized batch conversion to COCO format
    coco_dets = []
    N = dets.masks.shape[0]
    
    # Pre-allocate and batch process
    masks_np = dets.masks.numpy().astype(np.uint8)
    
    for i in range(N):
        rle = cocomask.encode(np.asfortranarray(masks_np[i]))
        coco_dets.append({
            "segmentation": rle,
            "score": float(dets.scores[i]) if hasattr(dets, "scores") else 1.0,
            "category_id": int(dets.labels[i]) if hasattr(dets, "labels") else 1,
        })

    # visualize
    vis_path = os.path.join(OUTPUT_DIR, "sam6d_results", "vis_ism.png")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    visualize_seg(rgb_pil, coco_dets, vis_path)

    return coco_dets, rgb_np

# ─── OPTIMIZED POSE‐ESTIMATION (PEM) ────────────────────────────────────────────
def get_test_data_optimized(raw_js, ds_cfg):
    # Filter and sort detections
    dets = sorted(
        [d for d in raw_js if d["score"] > DET_SCORE_THRESH],
        key=lambda d: d["score"], reverse=True
    )[:TOP_K]
    if not dets:
        raise RuntimeError("No ISM masks above threshold")

    # Use cached static data
    static_data = get_static_data()
    K = static_data['K']
    img = static_data['img']
    depth = static_data['depth']
    pts3 = static_data['pts3']
    
    # Use cached model points
    model_pts, rad = get_model_points(ds_cfg)

    # Pre-allocate lists for better memory management
    all_pts, all_rgb, all_choose, scores, kept = [], [], [], [], []
    
    # Batch process detections
    for d in dets:
        seg = d["segmentation"]; h,w = seg["size"]
        try: 
            rle = cocomask.frPyObjects(seg, h, w)
        except: 
            rle = seg
        mask = cocomask.decode(rle) & (depth > 0)
        if mask.sum() < 32: 
            continue

        y1,y2,x1,x2 = get_bbox(mask)
        m = mask[y1:y2, x1:x2]
        choose = m.flatten().nonzero()[0]

        cloud = pts3[y1:y2, x1:x2, :].reshape(-1,3)[choose]
        c0 = cloud.mean(0)
        keep_mask = np.linalg.norm(cloud-c0, axis=1) < rad*1.2
        if keep_mask.sum() < 4: 
            continue

        total = keep_mask.sum()
        idxs = np.random.choice(
            total,
            ds_cfg.n_sample_observed_point,
            replace=(total < ds_cfg.n_sample_observed_point),
        )
        choose = choose[keep_mask][idxs]
        cloud = cloud[keep_mask][idxs]

        patch = img[y1:y2, x1:x2][:,:,::-1]
        patch = cv2.resize(patch, (ds_cfg.img_size, ds_cfg.img_size), interpolation=cv2.INTER_LINEAR)
        if RGB_MASK_FLAG:
            patch = patch * (m[:,:,None] > 0).astype(np.uint8)
        patch_t = rgb_transform(patch)
        rc = get_resize_rgb_choose(choose, [y1,y2,x1,x2], ds_cfg.img_size)

        # Direct tensor creation on device to reduce CPU-GPU transfers
        all_pts.append(torch.FloatTensor(cloud).to(device))
        all_rgb.append(torch.FloatTensor(patch_t).to(device))
        all_choose.append(torch.LongTensor(rc).to(device))
        scores.append(d["score"])
        kept.append(d)

    if not all_pts:
        raise RuntimeError("No valid detections after filtering")

    n = len(all_pts)
    # Batch tensor operations
    data = {
        "pts": torch.stack(all_pts),
        "rgb": torch.stack(all_rgb),
        "rgb_choose": torch.stack(all_choose),
        "score": torch.FloatTensor(scores).to(device),
        "model": torch.FloatTensor(model_pts).unsqueeze(0).repeat(n,1,1).to(device),
        "K": torch.FloatTensor(K).unsqueeze(0).repeat(n,1,1).to(device),
    }
    return data, kept, model_pts

def run_pose_optimized(raw_js, rgb_np):
    # Use cached models and config
    pem, cfg = get_pem_model()
    
    data, kept, model_pts = get_test_data_optimized(raw_js, cfg.test_dataset)
    n = data["pts"].size(0)

    # Use cached templates
    all_tem, all_tem_pts, all_tem_choose = get_templates_cached(TEMPLATE_DIR, cfg.test_dataset)
    
    # Extract features from templates (cached computation when possible)
    with torch.no_grad():
        # get_obj_feats returns (pts_feats, rgb_feats)
        all_tem_pts_feat, all_tem_feat = pem.feature_extraction.get_obj_feats(
            all_tem, all_tem_pts, all_tem_choose
        )

    # Run inference
    with torch.no_grad():
        data["dense_po"] = all_tem_pts_feat.repeat(n,1,1)
        data["dense_fo"] = all_tem_feat.repeat(n,1,1)
        out = pem(data)

    # Process results
    if "pred_pose_score" in out:
        pose_scores = (out["pred_pose_score"] * out["score"]).cpu().numpy()
    else:
        pose_scores = out["score"].cpu().numpy()
    R = out["pred_R"].cpu().numpy()
    t = out["pred_t"].cpu().numpy() * 1000

    # Save results
    res_dir = os.path.join(OUTPUT_DIR, "sam6d_results")
    with open(os.path.join(res_dir, "detection_pem.json"), "w") as f:
        for i, d in enumerate(kept):
            d["score"] = float(pose_scores[i])
            d["R"] = R[i].tolist()
            d["t"] = t[i].tolist()
            # Handle RLE encoding
            seg = d.get("segmentation", {})
            cnt = seg.get("counts", None)
            if isinstance(cnt, (bytes, bytearray)):
               seg["counts"] = cnt.decode("ascii")
        json.dump(kept, f)

    # Prepare all detections for advanced visualization
    vis_detections = []
    for i, d in enumerate(kept):
        vis_detection = {
            'bbox': d.get('bbox', [0, 0, 100, 100]),
            'score': float(pose_scores[i]),
            'R': R[i].tolist(),
            't': t[i].tolist(),
            'segmentation': d.get('segmentation', {}),
            'category_id': d.get('category_id', 1)
        }
        vis_detections.append(vis_detection)
    
    # Use advanced visualization for all detections
    static_data = get_static_data()
    K_vis = static_data['K']
    
    visualize_pose_advanced(
        rgb_np,
        vis_detections,
        K_vis,
        os.path.join(res_dir, "vis_pem.png"),
    )
    
    # Print detection summary
    print(f"\nPose Estimation Results:")
    print("-" * 50)
    for i, (d, score) in enumerate(zip(kept, pose_scores)):
        euler_angles = rotation_matrix_to_euler(R[i])
        distance = np.linalg.norm(t[i])
        print(f"Detection {i}:")
        print(f"  Score: {score:.3f}")
        print(f"  Distance: {distance:.1f} mm")
        print(f"  Rotation (deg): Roll={euler_angles[0]:.1f}, Pitch={euler_angles[1]:.1f}, Yaw={euler_angles[2]:.1f}")
        print(f"  3D Position (mm): [{t[i][0]:.1f}, {t[i][1]:.1f}, {t[i][2]:.1f}]")
        print()

# ─── MULTI-OBJECT PROCESSING FUNCTIONS ─────────────────────────────────────────
def filter_detections_by_type(raw_js):
    """Filter and group detections by object type"""
    if not PROCESS_ALL_OBJECTS:
        # Original behavior - return top detections
        return sorted(
            [d for d in raw_js if d["score"] > DET_SCORE_THRESH],
            key=lambda d: d["score"], reverse=True
        )[:TOP_K]
    
    # Group by category_id and process multiple types
    grouped_dets = {}
    for d in raw_js:
        if d["score"] > DET_SCORE_THRESH:
            cat_id = d.get("category_id", 1)
            if cat_id not in grouped_dets:
                grouped_dets[cat_id] = []
            grouped_dets[cat_id].append(d)
    
    # Sort each group and take top N per category
    filtered_dets = []
    for cat_id, dets in grouped_dets.items():
        sorted_dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        filtered_dets.extend(sorted_dets[:MAX_OBJECTS_PER_TYPE])
    
    return filtered_dets

def get_object_config_by_category(category_id):
    """Get object configuration based on category ID"""
    for obj_name, config in MULTI_OBJECT_CONFIG.items():
        if config["category_id"] == category_id:
            return obj_name, config
    # Default to first object type if not found
    default_name = list(MULTI_OBJECT_CONFIG.keys())[0]
    return default_name, MULTI_OBJECT_CONFIG[default_name]

# ─── OPTIMIZED MAIN FUNCTION ────────────────────────────────────────────────────
def run_optimized():
    """Main optimized function that reuses models and cached data"""
    os.makedirs(os.path.join(OUTPUT_DIR, "sam6d_results"), exist_ok=True)

    start_time = time.time()
    seg_js, rgb_np = run_segmentation_optimized()
    end_seg = time.time()

    run_pose_optimized(seg_js, rgb_np)
    end_pose = time.time()

    print("SEG time: " + str(end_seg - start_time))
    print("POSE time: " + str(end_pose - end_seg))
    print("Total time: " + str(end_pose - start_time))

    # Create combined visualization
    im1 = Image.open(os.path.join(OUTPUT_DIR, "sam6d_results", "vis_ism.png"))
    im2 = Image.open(os.path.join(OUTPUT_DIR, "sam6d_results", "vis_pem.png"))
    combo = Image.new(
        "RGB",
        (im1.width + im2.width, max(im1.height, im2.height))
    )
    combo.paste(im1, (0,0))
    combo.paste(im2, (im1.width,0))
    combo.save(os.path.join(OUTPUT_DIR, "sam6d_results", "vis_combined.png"))

# ─── BACKWARD COMPATIBILITY ─────────────────────────────────────────────────────
# Keep original functions for compatibility
def run_segmentation():
    return run_segmentation_optimized()

def run_pose(raw_js, rgb_np):
    return run_pose_optimized(raw_js, rgb_np)

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run optimized version
    run_optimized()
    
    # For multiple iterations, just call run_optimized() again
    # The models and static data will be reused automatically
