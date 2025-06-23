#!/usr/bin/env python3
"""
RealSense D435i Camera Capture Script
Captures one frame of RGB and depth images and saves them as PNG files.
Also prints camera intrinsics to console.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from datetime import datetime

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    print(f"Connected RealSense Device: {device.get_info(rs.camera_info.name)}")
    print(f"Product Line: {device_product_line}")
    print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
    print("-" * 50)
    
    # Configure streams
    # For D435i, typical resolutions are 640x480, 848x480, 1280x720
    width, height = 1280, 720
    fps = 30
    
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    try:
        # Start streaming
        print("Starting camera pipeline...")
        pipeline.start(config)
        
        # Get camera intrinsics
        profile = pipeline.get_active_profile()
        
        # Get depth and color stream profiles
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        
        # Get intrinsics
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        
        # Print camera intrinsics
        print("DEPTH CAMERA INTRINSICS:")
        print(f"  Resolution: {depth_intrinsics.width} x {depth_intrinsics.height}")
        print(f"  Principal Point: ({depth_intrinsics.ppx:.2f}, {depth_intrinsics.ppy:.2f})")
        print(f"  Focal Length: ({depth_intrinsics.fx:.2f}, {depth_intrinsics.fy:.2f})")
        print(f"  Distortion Model: {depth_intrinsics.model}")
        print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")
        print()
        
        print("COLOR CAMERA INTRINSICS:")
        print(f"  Resolution: {color_intrinsics.width} x {color_intrinsics.height}")
        print(f"  Principal Point: ({color_intrinsics.ppx:.2f}, {color_intrinsics.ppy:.2f})")
        print(f"  Focal Length: ({color_intrinsics.fx:.2f}, {color_intrinsics.fy:.2f})")
        print(f"  Distortion Model: {color_intrinsics.model}")
        print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")
        print()
        
        # Create intrinsics matrix in standard format
        color_K = [
            color_intrinsics.fx, 0.0, color_intrinsics.ppx,
            0.0, color_intrinsics.fy, color_intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        depth_K = [
            depth_intrinsics.fx, 0.0, depth_intrinsics.ppx,
            0.0, depth_intrinsics.fy, depth_intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        print("COLOR CAMERA MATRIX (3x3 format):")
        print(f"[{color_K[0]:8.2f}, {color_K[1]:8.2f}, {color_K[2]:8.2f}]")
        print(f"[{color_K[3]:8.2f}, {color_K[4]:8.2f}, {color_K[5]:8.2f}]")
        print(f"[{color_K[6]:8.2f}, {color_K[7]:8.2f}, {color_K[8]:8.2f}]")
        print()
        
        # Wait for a coherent pair of frames: depth and color
        print("Waiting for frames...")
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Error: Could not acquire frames")
            return
            
        print("Frames acquired successfully!")
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get depth scale (to convert to meters)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {depth_scale} (meters per unit)")
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images
        rgb_filename = f"rgb_{timestamp}.png"
        depth_filename = f"depth_{timestamp}.png"
        
        # Save RGB image
        cv2.imwrite(rgb_filename, color_image)
        print(f"RGB image saved as: {rgb_filename}")
        
        # Save depth image (16-bit PNG to preserve depth values)
        cv2.imwrite(depth_filename, depth_image)
        print(f"Depth image saved as: {depth_filename}")
        
        # Save camera intrinsics in the same format as SAM-6D example
        camera_data = {
            "cam_K": color_K,
            "depth_scale": depth_scale
        }
        
        # Save in standard camera.json format
        camera_filename = "camera.json"
        with open(camera_filename, 'w') as f:
            json.dump(camera_data, f)
        print(f"Camera intrinsics saved as: {camera_filename}")
        print(f"Format matches SAM-6D expected format:")
        print(f"  cam_K: {color_K}")
        print(f"  depth_scale: {depth_scale}")
        
        # Also save detailed intrinsics with timestamp for reference
        detailed_camera_data = {
            "timestamp": timestamp,
            "device_info": {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version)
            },
            "color_intrinsics": {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy,
                "model": str(color_intrinsics.model),
                "coeffs": list(color_intrinsics.coeffs),
                "cam_K": color_K
            },
            "depth_intrinsics": {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "model": str(depth_intrinsics.model),
                "coeffs": list(depth_intrinsics.coeffs),
                "cam_K": depth_K
            },
            "depth_scale": depth_scale
        }
        
        detailed_camera_filename = f"camera_intrinsics_detailed_{timestamp}.json"
        with open(detailed_camera_filename, 'w') as f:
            json.dump(detailed_camera_data, f, indent=2)
        print(f"Detailed camera intrinsics saved as: {detailed_camera_filename}")
        
        # Display some statistics about the captured images
        print(f"\nImage Statistics:")
        print(f"RGB Image Shape: {color_image.shape}")
        print(f"Depth Image Shape: {depth_image.shape}")
        print(f"Depth Image Min/Max: {depth_image.min()} / {depth_image.max()} (raw units)")
        print(f"Depth Range: {depth_image.min() * depth_scale:.3f}m to {depth_image.max() * depth_scale:.3f}m")
        
        print(f"\nCapture completed successfully!")
        print(f"Files saved:")
        print(f"  - {rgb_filename}")
        print(f"  - {depth_filename}")
        print(f"  - camera.json (SAM-6D compatible format)")
        print(f"  - {detailed_camera_filename} (detailed reference)")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        # Stop streaming
        pipeline.stop()
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()
