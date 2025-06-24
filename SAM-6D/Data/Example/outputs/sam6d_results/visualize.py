import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# ===== GLOBAL VISUALIZATION FLAGS =====
SHOW_2D_BBOX = True          # Show 2D bounding boxes
SHOW_3D_BBOX = False          # Show 3D bounding boxes with pose
SHOW_POSE_AXES = True        # Show coordinate axes at object center
SHOW_SEGMENTATION = True     # Show segmentation masks
SHOW_POSE_INFO = False        # Show pose information text
SHOW_SCORE_INFO = True       # Show detection scores
SHOW_DISTANCE_INFO = False    # Show distance information
SHOW_OBJECT_ID = True        # Show object IDs

# Filtering parameters
SCORE_THRESHOLD = 0.0        # Only show detections with score > this value
SHOW_ONLY_HIGHEST_SCORE = True  # Show only the object with highest score
SHOW_SPECIFIC_OBJECT_ID = None   # Show only specific object ID (0, 1, 2, etc.) or None for all
                                 # Note: This overrides SHOW_ONLY_HIGHEST_SCORE if set
                                 # To use: set to integer ID (e.g., 0 for first object, 1 for second, etc.)

# Visualization parameters
MASK_ALPHA = 0.3             # Transparency for masks
TEXT_COLOR = (255, 255, 0)   # Yellow for text
BBOX_2D_COLOR = (255, 255, 0) # Yellow for 2D boxes
POSE_AXIS_LENGTH = 0.05      # Length of pose axes

# ===== FILE SELECTION =====
# Specify which detection file to use:
# "detection_ism.json" - for 2D instance segmentation results
# "detection_pem.json" - for 6D pose estimation results
JSON_FILE = "detection_pem.json"  # Change this to select which file to load
# =======================================

def decode_rle_mask(rle, height, width):
    """Decode RLE mask to binary mask"""
    counts = rle['counts']
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

def print_flags_status():
    """Print current visualization flags status"""
    print("Current Visualization Settings:")
    print("-" * 40)
    print(f"Detection Type:       {detection_type}")
    print(f"JSON File:            {json_file}")
    print(f"Score Threshold:      {SCORE_THRESHOLD}")
    print(f"Show Only Highest:    {'ON' if SHOW_ONLY_HIGHEST_SCORE else 'OFF'}")
    print(f"Show Specific ID:     {SHOW_SPECIFIC_OBJECT_ID if SHOW_SPECIFIC_OBJECT_ID is not None else 'OFF'}")
    print(f"2D Bounding Boxes:    {'ON' if SHOW_2D_BBOX else 'OFF'}")
    print(f"3D Bounding Boxes:    {'ON' if SHOW_3D_BBOX else 'OFF'}")
    print(f"Pose Axes:            {'ON' if SHOW_POSE_AXES else 'OFF'}")
    print(f"Segmentation Masks:   {'ON' if SHOW_SEGMENTATION else 'OFF'}")
    print(f"Pose Information:     {'ON' if SHOW_POSE_INFO else 'OFF'}")
    print(f"Score Information:    {'ON' if SHOW_SCORE_INFO else 'OFF'}")
    print(f"Distance Information: {'ON' if SHOW_DISTANCE_INFO else 'OFF'}")
    print(f"Object IDs:           {'ON' if SHOW_OBJECT_ID else 'OFF'}")
    print("-" * 40)

# Load image
import os
img_path = os.environ.get("RGB_PATH")
if img_path is None:
  raise ValueError("RGB_PATH environment variable is not set.")
img = cv2.imread(img_path)
if img is None:
  raise ValueError(f"Failed to load image from path: {img_path}")
img_with_masks = img.copy()  # Create a copy for mask visualization

# Load JSON file and auto-configure visualization based on file type
json_file = JSON_FILE

# Check if the specified file exists
if not os.path.exists(json_file):
    raise FileNotFoundError(f"Specified JSON file not found: {json_file}")

# Auto-configure visualization settings based on file type
if json_file.endswith("_pem.json"):
    detection_type = "PEM"
    print(f"Loading PEM detection file: {json_file}")
    print("Auto-configuring for 6D Pose Estimation visualization...")
    # Override visualization flags for PEM (3D pose estimation)
    SHOW_2D_BBOX = True
    SHOW_3D_BBOX = True
    SHOW_POSE_AXES = True
    SHOW_SEGMENTATION = False
    SHOW_POSE_INFO = True
    SHOW_SCORE_INFO = True
    SHOW_DISTANCE_INFO = True
    SHOW_OBJECT_ID = True
elif json_file.endswith("_ism.json"):
    detection_type = "ISM"
    print(f"Loading ISM detection file: {json_file}")
    print("Auto-configuring for 2D Instance Segmentation visualization...")
    # Override visualization flags for ISM (2D instance segmentation)
    SHOW_2D_BBOX = True
    SHOW_3D_BBOX = False
    SHOW_POSE_AXES = False
    SHOW_SEGMENTATION = True
    SHOW_POSE_INFO = False
    SHOW_SCORE_INFO = True
    SHOW_DISTANCE_INFO = False
    SHOW_OBJECT_ID = True
else:
    detection_type = "UNKNOWN"
    print(f"Loading detection file: {json_file} (type unknown - using current settings)")

# Load the specified JSON file
with open(json_file) as f:
    preds = json.load(f)

print(f"Detection type: {detection_type}")
print(f"Loaded {len(preds)} predictions")

# Filter predictions based on settings
if SHOW_SPECIFIC_OBJECT_ID is not None and len(preds) > 0:
    # Show only specific object ID (this overrides SHOW_ONLY_HIGHEST_SCORE)
    if SHOW_SPECIFIC_OBJECT_ID < len(preds):
        specific_obj = preds[SHOW_SPECIFIC_OBJECT_ID]
        preds = [specific_obj]  # Keep only the specified object
        print(f"Filtering: Showing only object ID {SHOW_SPECIFIC_OBJECT_ID} (score: {specific_obj['score']:.3f})")
    else:
        print(f"Warning: Object ID {SHOW_SPECIFIC_OBJECT_ID} not found. Available IDs: 0-{len(preds)-1}")
        preds = []  # No valid object to show
elif SHOW_ONLY_HIGHEST_SCORE and len(preds) > 0:
    # Find the object with the highest score
    highest_score_obj = max(preds, key=lambda x: x['score'])
    preds = [highest_score_obj]  # Keep only the highest scoring object
    print(f"Filtering: Showing only highest score object (score: {highest_score_obj['score']:.3f})")
else:
    print(f"Total detections loaded: {len(preds)}")

# Print current settings after auto-detection and filtering
print_flags_status()

# Load camera intrinsics from camera.json
with open("../../camera_intrinsics/camera.json") as f:
    cam_data = json.load(f)
    cam_K = cam_data['cam_K']
    K = np.array([
        [cam_K[0], cam_K[1], cam_K[2]],
        [cam_K[3], cam_K[4], cam_K[5]],
        [cam_K[6], cam_K[7], cam_K[8]]
    ])

def create_3d_bbox_from_2d_center(bbox_2d, depth_est, K, aspect_ratio=1.0):
    """Create 3D bounding box corners based on 2D bbox center and estimated depth"""
    x, y, w, h = bbox_2d
    
    # Calculate 2D bounding box center
    center_2d_x = x + w / 2
    center_2d_y = y + h / 2
    
    # Back-project 2D center to 3D space at given depth
    # Using inverse camera projection: P_3D = depth * K^(-1) * [u, v, 1]^T
    K_inv = np.linalg.inv(K)
    center_2d_homogeneous = np.array([center_2d_x, center_2d_y, 1.0])
    center_3d = depth_est * (K_inv @ center_2d_homogeneous)  # 3D center in camera coordinates
    
    # Estimate object dimensions based on 2D bbox size and depth
    # Convert pixel size to approximate world size using similar triangles
    obj_width = w * depth_est / K[0, 0]   # Using focal length fx
    obj_height = h * depth_est / K[1, 1]  # Using focal length fy
    obj_depth = min(obj_width, obj_height) * aspect_ratio  # Assume depth proportional to other dims
    
    # Define 3D box corners relative to the calculated 3D center
    half_w, half_h, half_d = obj_width/2, obj_height/2, obj_depth/2
    
    # Create box corners centered at origin, then translate to 3D center
    box_3D_local = np.array([
        [-half_w, -half_h, -half_d],  # bottom face
        [ half_w, -half_h, -half_d],
        [ half_w,  half_h, -half_d],
        [-half_w,  half_h, -half_d],
        [-half_w, -half_h,  half_d],  # top face
        [ half_w, -half_h,  half_d],
        [ half_w,  half_h,  half_d],
        [-half_w,  half_h,  half_d],
    ]).T  # shape: (3, 8)
    
    # Translate box to 3D center position
    box_3D = box_3D_local + center_3d.reshape(3, 1)
    
    return box_3D, center_3d

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (in degrees)"""
    # Extract Euler angles (ZYX convention)
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

for i, obj in enumerate(preds):
    # Extract data from detection result
    bbox_2d = obj['bbox']  # [x, y, width, height]
    score = obj['score']
    if SHOW_3D_BBOX:
      R = np.array(obj['R'])  # 3x3
      t = np.array(obj['t']).reshape(3, 1)  # 3x1
    segmentation = obj['segmentation']
    
    # Apply score threshold filter
    if score < SCORE_THRESHOLD:
        continue  # Skip this detection if score is below threshold
    
    # Decode and overlay segmentation mask
    if SHOW_SEGMENTATION and 'counts' in segmentation and 'size' in segmentation:
        mask = decode_rle_mask(segmentation, segmentation['size'][0], segmentation['size'][1])
        
        # Create colored mask overlay
        color = np.array([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
        mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_colored[mask == 1] = color
        
        # Blend mask with image
        alpha = MASK_ALPHA
        img_with_masks = cv2.addWeighted(img_with_masks, 1-alpha, mask_colored, alpha, 0)
    
    # Draw 2D bounding box
    x, y, w, h = bbox_2d

    if SHOW_2D_BBOX:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), BBOX_2D_COLOR, 2)
        cv2.rectangle(img_with_masks, (int(x), int(y)), (int(x + w), int(y + h)), BBOX_2D_COLOR, 2)
    
    # Draw detection score and pose information
    if SHOW_3D_BBOX:
      euler_angles = rotation_matrix_to_euler(R)
      distance = np.linalg.norm(t)
    
    if SHOW_SCORE_INFO:
        cv2.putText(img, f'Score: {score:.3f}', (int(x), int(y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        cv2.putText(img_with_masks, f'Score: {score:.3f}', (int(x), int(y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Add distance information
    if SHOW_DISTANCE_INFO:
        cv2.putText(img, f'Dist: {distance:.0f}mm', (int(x), int(y - 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        cv2.putText(img_with_masks, f'Dist: {distance:.0f}mm', (int(x), int(y - 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    # Add pose information
    if SHOW_POSE_INFO:
        cv2.putText(img, f'Rot: ({euler_angles[0]:.1f}°,{euler_angles[1]:.1f}°,{euler_angles[2]:.1f}°)', 
              (int(x), int(y - 40)), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        cv2.putText(img_with_masks, f'Rot: ({euler_angles[0]:.1f}°,{euler_angles[1]:.1f}°,{euler_angles[2]:.1f}°)', 
              (int(x), int(y - 40)), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    # Draw detection ID
    if SHOW_OBJECT_ID:
        cv2.putText(img, f'ID: {i}', (int(x), int(y + h + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        cv2.putText(img_with_masks, f'ID: {i}', (int(x), int(y + h + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    # Calculate 3D bounding box dimensions (needed for both 3D bbox and pose axes)
    if SHOW_3D_BBOX:
      depth_estimate = np.linalg.norm(t)  # Use distance from camera as depth estimate
      box_3D, center_3d= create_3d_bbox_from_2d_center(bbox_2d, depth_estimate,K , aspect_ratio=0.8)
    
      # Get bounding box dimensions for axis scaling
      x, y, w, h = bbox_2d
      obj_width = w * depth_estimate / 1000.0
      obj_height = h * depth_estimate / 1000.0
      obj_depth = min(obj_width, obj_height) * 0.8

    if SHOW_3D_BBOX:
        # Transform 3D box to camera coordinates (apply rotation and translation)
        box_cam = R @ box_3D + t  # shape: (3, 8)
        proj = K @ box_cam
        
        # Check if points are in front of camera (positive Z)
        valid_depth = proj[2] > 0
        if not np.all(valid_depth):
            print(f"Warning: Object {i} has points behind camera, skipping 3D visualization")
            continue
        
        proj = proj[:2] / proj[2]  # normalize by depth
        proj = proj.T.astype(int)

    # Draw lines between box corners with different colors for each object
    color_offset = i * 50 % 255
    box_color = (255 - color_offset, color_offset, 128)
    edge_thickness = 3
    
    # Draw 3D box on both images with better visualization
    if SHOW_3D_BBOX:
        for target_img in [img, img_with_masks]:
            # Draw bottom face (darker color)
            bottom_color = tuple(int(c * 0.7) for c in box_color)
            for j in range(4):
                cv2.line(target_img, tuple(proj[j]), tuple(proj[(j + 1) % 4]), bottom_color, edge_thickness)
            
            # Draw top face (brighter color)  
            top_color = box_color
            for j in range(4, 8):
                cv2.line(target_img, tuple(proj[j]), tuple(proj[4 + (j + 1) % 4]), top_color, edge_thickness)
            
            # Draw vertical edges (medium color)
            vertical_color = tuple(int(c * 0.85) for c in box_color)
            for j in range(4):
                cv2.line(target_img, tuple(proj[j]), tuple(proj[j + 4]), vertical_color, edge_thickness)
            
            # Draw center point
            x, y, w, h = bbox_2d
            center_2d = (int(x + w / 2), int(y + h / 2))
            cv2.circle(target_img, center_2d, 4, (255, 255, 255), -1)
            cv2.circle(target_img, (center_2d), 5, box_color, 2)

    # Draw coordinate axes to show object orientation
    if SHOW_POSE_AXES:
        # Set axis lengths to reach the edges of the bounding box
        x_axis_length = obj_width / 2    # Extends to edge of bbox in X direction
        y_axis_length = obj_height / 2   # Extends to edge of bbox in Y direction  
        z_axis_length = obj_depth / 2    # Extends to edge of bbox in Z direction
        
        axes = np.float32([
            [0, 0, 0],                          # Origin
            [x_axis_length, 0, 0],              # X axis (red) - extends to bbox edge
            [0, y_axis_length, 0],              # Y axis (green) - extends to bbox edge
            [0, 0, z_axis_length]               # Z axis (blue) - extends to bbox edge
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
            
            # Draw axes on both images with labels
            for target_img in [img, img_with_masks]:
                # X axis - Red (extends to edge of bounding box)
                cv2.arrowedLine(target_img, origin, x_end, (0, 0, 255), 4, tipLength=0.2)
                cv2.putText(target_img, 'X', (x_end[0] + 5, x_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Y axis - Green (extends to edge of bounding box)
                cv2.arrowedLine(target_img, origin, y_end, (0, 255, 0), 4, tipLength=0.2)
                cv2.putText(target_img, 'Y', (y_end[0] + 5, y_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Z axis - Blue (extends to edge of bounding box)
                cv2.arrowedLine(target_img, origin, z_end, (255, 0, 0), 4, tipLength=0.2)
                cv2.putText(target_img, 'Z', (z_end[0] + 5, z_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show the results
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Generate dynamic titles based on active flags
title_parts = []
if SHOW_2D_BBOX: title_parts.append("2D Boxes")
if SHOW_3D_BBOX: title_parts.append("3D Boxes")
if SHOW_POSE_AXES: title_parts.append("Pose Axes")

title1 = "SAM-6D: " + " + ".join(title_parts) if title_parts else "SAM-6D Results"

# Display image with bounding boxes and 3D poses
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title(title1)
axes[0].axis("off")

# Display image with segmentation masks
mask_title_parts = []
if SHOW_SEGMENTATION: mask_title_parts.append("Segmentation")
if title_parts: mask_title_parts.extend(title_parts)

title2 = "SAM-6D: " + " + ".join(mask_title_parts) if mask_title_parts else "SAM-6D Results"

axes[1].imshow(cv2.cvtColor(img_with_masks, cv2.COLOR_BGR2RGB))
axes[1].set_title(title2)
axes[1].axis("off")

plt.tight_layout()
plt.show()

# Also save the output images
cv2.imwrite("visualization_poses.png", img)
cv2.imwrite("visualization_masks.png", img_with_masks)

# Count filtered vs total detections
# Note: preds may already be filtered by SHOW_ONLY_HIGHEST_SCORE or SHOW_SPECIFIC_OBJECT_ID
if SHOW_SPECIFIC_OBJECT_ID is not None:
    total_detections = 1  # Only one specific object if ID filtering is on
    filtered_detections = len([obj for obj in preds if obj['score'] >= SCORE_THRESHOLD])
    print(f"Pose visualization saved to: visualization_poses.png")
    print(f"Mask visualization saved to: visualization_masks.png")
    print(f"Showing only specific object ID: {SHOW_SPECIFIC_OBJECT_ID}")
    print(f"Specific object meets threshold: {'YES' if filtered_detections > 0 else 'NO'}")
elif SHOW_ONLY_HIGHEST_SCORE:
    total_detections = 1  # Only one object if highest score filtering is on
    filtered_detections = len([obj for obj in preds if obj['score'] >= SCORE_THRESHOLD])
    print(f"Pose visualization saved to: visualization_poses.png")
    print(f"Mask visualization saved to: visualization_masks.png")
    print(f"Showing only highest score object")
    print(f"Highest score object meets threshold: {'YES' if filtered_detections > 0 else 'NO'}")
else:
    total_detections = len(preds)
    filtered_detections = len([obj for obj in preds if obj['score'] >= SCORE_THRESHOLD])
    print(f"Pose visualization saved to: visualization_poses.png")
    print(f"Mask visualization saved to: visualization_masks.png")
    print(f"Total detections: {total_detections}")
    print(f"Detections above threshold ({SCORE_THRESHOLD}): {filtered_detections}")
    print(f"Detections filtered out: {total_detections - filtered_detections}")

# Print detection summary with pose information
if SHOW_SPECIFIC_OBJECT_ID is not None:
    print(f"\n{detection_type} Detection Summary for Specific Object ID {SHOW_SPECIFIC_OBJECT_ID}:")
elif SHOW_ONLY_HIGHEST_SCORE:
    print(f"\n{detection_type} Detection Summary for Highest Score Object:")
else:
    print(f"\n{detection_type} Detection Summary:")
    print(f"(Only showing detections with score >= {SCORE_THRESHOLD})")
print("-" * 80)
for i, obj in enumerate(preds):
    bbox = obj['bbox']
    score = obj['score']
    if SHOW_3D_BBOX:
      R = np.array(obj['R'])
      t = np.array(obj['t'])
    
    # Apply score threshold filter for summary too
    if score < SCORE_THRESHOLD:
        continue
    
    # Convert rotation to Euler angles for readability
    if SHOW_3D_BBOX:
      euler_angles = rotation_matrix_to_euler(R)
      distance = np.linalg.norm(t)
      print(f"  Distance: {distance:.1f} mm")
      
    print(f"Detection {i}:")
    print(f"  Score: {score:.3f}")
    print(f"  2D BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    if SHOW_3D_BBOX:
      print(f"  Rotation (deg): Roll={euler_angles[0]:.1f}, Pitch={euler_angles[1]:.1f}, Yaw={euler_angles[2]:.1f}")
      print(f"  3D Position (mm): [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]")

    print()