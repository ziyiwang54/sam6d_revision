import cv2

depth_original = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
depth_check = cv2.imread("depth_20250616_173232.png", cv2.IMREAD_UNCHANGED)
rgb_original = cv2.imread("rgb.png", cv2.IMREAD_COLOR)
rgb_check = cv2.imread("rgb_20250616_173232.png", cv2.IMREAD_COLOR)
print(rgb_check.shape)

# Check for dimensions
print("Original Depth Image Shape:", depth_original.shape)
print("Check Depth Image Shape:", depth_check.shape)
if depth_original.shape != depth_check.shape:
    print("Depth images have different dimensions.")
else:
    # Check for pixel values
    if (depth_original == depth_check).all():
        print("Depth images are identical.")
    else:
        print("Depth images differ in pixel values.")
        # Optionally, you can save the difference image
        difference = cv2.absdiff(depth_original, depth_check)
        cv2.imwrite("depth_difference.png", difference)
        print("Difference image saved as 'depth_difference.png'.")  

