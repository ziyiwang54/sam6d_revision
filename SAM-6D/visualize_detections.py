import json
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
import distinctipy

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def decode_rle(rle, size):
    """Decode RLE encoded mask. Handles uncompressed RLE (list) manually."""
    if isinstance(rle, dict):
        counts = rle['counts']
        if isinstance(counts, list):
            # Manual decode for uncompressed RLE
            h, w = size
            mask = np.zeros(h * w, dtype=np.uint8)
            idx = 0
            val = 0
            for c in counts:
                mask[idx:idx+c] = val
                idx += c
                val = 1 - val
            mask = mask.reshape((h, w), order='F')
            return mask
        else:
            # Try pycocotools for compressed RLE
            rle = {'counts': counts, 'size': size}
            mask = coco_mask.decode([rle])[:, :, 0]
            return mask
    else:
        # Try pycocotools for compressed RLE
        mask = coco_mask.decode([rle])[:, :, 0]
        return mask

def visualize_detections(image_path, ism_path, pem_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load detection results
    ism_detections = load_json_file(ism_path)
    pem_detections = load_json_file(pem_path)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image with ISM detections
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot image with ISM detections
    plt.subplot(132)
    plt.imshow(image)
    colors = distinctipy.get_colors(len(ism_detections))
    
    for i, det in enumerate(ism_detections):
        # Draw bounding box
        x, y, w, h = det['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color=colors[i], linewidth=2)
        plt.gca().add_patch(rect)
        
        # Draw segmentation mask
        mask = decode_rle(det['segmentation'], det['segmentation']['size'])
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask > 0] = [int(c * 255) for c in colors[i]]
        plt.imshow(mask_overlay, alpha=0.3)
        
        # Add score
        plt.text(x, y-5, f"Score: {det['score']:.2f}", 
                color='white', fontsize=8, 
                bbox=dict(facecolor=colors[i], alpha=0.5))
    
    plt.title('ISM Detections')
    plt.axis('off')
    
    # Plot image with PEM detections
    plt.subplot(133)
    plt.imshow(image)
    
    for i, det in enumerate(pem_detections):
        # Draw bounding box
        x, y, w, h = det['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color=colors[i], linewidth=2)
        plt.gca().add_patch(rect)
        
        # Draw segmentation mask
        mask = decode_rle(det['segmentation'], det['segmentation']['size'])
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask > 0] = [int(c * 255) for c in colors[i]]
        plt.imshow(mask_overlay, alpha=0.3)
        
        # Add score and pose info
        pose_info = f"Score: {det['score']:.2f}\nR: {det['R'][0][0]:.2f}, {det['R'][0][1]:.2f}\nt: {det['t'][0]:.1f}, {det['t'][1]:.1f}"
        plt.text(x, y-5, pose_info, 
                color='white', fontsize=8, 
                bbox=dict(facecolor=colors[i], alpha=0.5))
    
    plt.title('PEM Detections with Pose')
    plt.axis('off')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define paths
    image_path = "Data/Example/rgb.png"
    ism_path = "Data/Example/outputs/sam6d_results/detection_ism.json"
    pem_path = "Data/Example/outputs/sam6d_results/detection_pem.json"
    output_path = "Data/Example/outputs/sam6d_results/visualize_bboxes.png"
    
    # Create visualization
    visualize_detections(image_path, ism_path, pem_path, output_path)
    print(f"Visualization saved to {output_path}") 