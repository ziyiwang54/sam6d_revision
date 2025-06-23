# Debug Flag Usage Guide

The SAM-6D pose estimation script now supports a `--debug` flag to control verbose debug output and debug image exports.

## Usage

### Without Debug (Default - Minimal Output)
```bash
python run_inference_custom.py \
    --output_dir /path/to/output \
    --cad_path /path/to/model.ply \
    --rgb_path /path/to/rgb.png \
    --depth_path /path/to/depth.png \
    --cam_path /path/to/camera.json \
    --seg_path /path/to/segmentation.json
```

### With Debug (Verbose Output + Debug Images)
```bash
python run_inference_custom.py \
    --debug \
    --output_dir /path/to/output \
    --cad_path /path/to/model.ply \
    --rgb_path /path/to/rgb.png \
    --depth_path /path/to/depth.png \
    --cam_path /path/to/camera.json \
    --seg_path /path/to/segmentation.json
```

## What Changes with --debug

### Debug Output Includes:
- Camera intrinsics details
- RGB and depth image loading details
- Point cloud generation details
- CAD model analysis (extent, radius, thresholds)
- Per-detection processing details:
  - Mask size and valid pixels
  - Depth values within mask (min, max, mean, std)
  - Bounding box analysis
  - Point cloud analysis (center, extent, distances)
  - Adaptive radius calculations
  - Filtering results and reasons
- Depth usage visualizations
- Color coding explanations

### Debug Images Exported (only with --debug):
- `debug_bbox/detection_X_bbox_depth.png` - Depth within bounding box
- `debug_bbox/detection_X_bbox_with_mask.png` - Depth with mask overlay
- `debug_bbox/detection_X_depth_with_bbox.png` - Full depth with bounding box
- `debug_bbox/detection_X_depth_with_radius.png` - Depth with adaptive radius visualization
- `sam6d_results/depth_original.png` - Original depth image
- `sam6d_results/depth_with_segmentation.png` - Depth with segmentation overlay
- `sam6d_results/depth_used_regions.png` - Only used depth regions
- `sam6d_results/segmentation_masks.png` - Segmentation masks visualization

### Without --debug:
- Only essential progress messages are shown
- No debug images are exported
- Significantly reduced console output
- Faster execution

## Benefits

1. **Development/Debugging**: Use `--debug` to understand why detections fail or pass
2. **Production**: Use without `--debug` for clean, minimal output
3. **Performance**: Debug mode is slower due to image processing and file I/O
4. **Storage**: Debug images can take significant disk space

## Color Coding in Debug Images

- **Red**: Primary detection mask
- **Yellow/Cyan**: Points actually used for pose estimation
- **Blue**: Relaxed criteria detection mask  
- **White**: Points used with relaxed criteria
- **Green**: Bounding boxes
- **Yellow circles**: Adaptive radius
- **Cyan circles**: Relaxed radius
- **Magenta**: Object center
