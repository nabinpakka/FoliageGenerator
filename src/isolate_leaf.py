# import cv2
# import numpy as np
# import os

# def isolate_overlapping_leaves(image_path, output_dir):
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Could not read the image")
    
#     # Convert to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     # Create mask for green leaves
#     lower_green = np.array([35, 30, 30])
#     upper_green = np.array([85, 255, 255])
#     green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
#     # Noise removal
#     kernel = np.ones((3,3), np.uint8)
#     clean_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Distance transform
#     dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
#     # Normalize distance transform for better visualization and thresholding
#     cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
#     # Threshold to get markers for watershed
#     _, sure_fg = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
#     sure_fg = np.uint8(sure_fg)
    
#     # Find background
#     sure_bg = cv2.dilate(clean_mask, kernel, iterations=3)
    
#     # Find unknown region
#     unknown = cv2.subtract(sure_bg, sure_fg)
    
#     # Marker labelling
#     _, markers = cv2.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown == 255] = 0
    
#     # Apply watershed
#     markers = cv2.watershed(img, markers)
    
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Process each segmented region
#     unique_markers = np.unique(markers)
#     leaf_count = 0
#     min_area = 1000  # Minimum area to consider as a leaf
    
#     for marker in unique_markers:
#         if marker <= 1:  # Skip background and boundary markers
#             continue
            
#         # Create mask for this marker
#         leaf_mask = np.zeros_like(green_mask)
#         leaf_mask[markers == marker] = 255
        
#         # Check area
#         area = np.sum(leaf_mask > 0)
#         if area < min_area:
#             continue
            
#         # Refine leaf mask
#         leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        
#         # Create transparent background
#         rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#         rgba[:, :, 3] = leaf_mask
        
#         # Get bounding rectangle
#         contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             continue
            
#         x, y, w, h = cv2.boundingRect(contours[0])
#         # Add padding
#         padding = 10
#         x, y = max(0, x - padding), max(0, y - padding)
#         w, h = min(img.shape[1] - x, w + 2*padding), min(img.shape[0] - y, h + 2*padding)
        
#         # Crop the image
#         leaf_image = rgba[y:y+h, x:x+w]
        
#         # Save the isolated leaf
#         output_path = os.path.join(output_dir, f'leaf_{leaf_count+1}.png')
#         cv2.imwrite(output_path, leaf_image)
#         leaf_count += 1
    
#     return leaf_count

# # Additional utility function for visualization
# def create_watershed_visualization(markers, img):
#     # Create a copy of the image
#     vis_img = img.copy()
#     # Mark watershed boundaries
#     vis_img[markers == -1] = [0, 0, 255]  # Red color for boundaries
#     return vis_img

# if __name__ == "__main__":
#     input_image = "/home/nabin/Documents/DiseaseClassification/src/output_images/images_8090%_healthy_low/train/downey_mildew/2_8_31.png"  # Replace with your image path
#     output_directory = "isolated_leaves"     # Output directory name
    
#     try:
#         num_leaves = isolate_overlapping_leaves(input_image, output_directory)
#         print(f"Successfully isolated {num_leaves} leaves. Check the '{output_directory}' folder.")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")


import cv2
import numpy as np
import os
import random

def generate_random_colors(n):
    """Generate n random distinct colors"""
    colors = []
    for _ in range(n):
        color = (random.randint(0, 255), 
                random.randint(0, 255), 
                random.randint(0, 255))
        colors.append(color)
    return colors

def enhance_leaf_edges(img):
    """Enhance edges while preserving color information"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l_channel)
    
    # Replace L channel and convert back
    lab[:,:,0] = enhanced_l
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def isolate_and_visualize_leaves(image_path, output_dir):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    original = img.copy()
    enhanced = enhance_leaf_edges(img)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Create color masks for different green shades
    lower_green1 = np.array([25, 20, 20])
    upper_green1 = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # Save initial color mask
    cv2.imwrite(os.path.join(output_dir, '1_color_mask.png'), green_mask)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine color mask with edges
    combined_mask = cv2.bitwise_or(green_mask, edges)
    cv2.imwrite(os.path.join(output_dir, '2_combined_mask.png'), combined_mask)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in cleaned mask
    contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask for visualization
    colored_masks = np.zeros_like(img)
    contour_img = img.copy()
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Process each contour
    min_area = 500
    colors = generate_random_colors(len(contours))
    leaf_count = 0
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Create mask for this contour
        leaf_mask = np.zeros_like(gray)
        cv2.drawContours(leaf_mask, [contour], -1, (255), -1)
        
        # Refine mask using color information
        color_region = cv2.bitwise_and(green_mask, leaf_mask)
        if np.sum(color_region) / 255 < area * 0.3:  # Skip if less than 30% is green
            continue
        
        # Draw contour
        cv2.drawContours(contour_img, [contour], -1, colors[idx], 2)
        
        # Add to colored visualization
        colored_masks[leaf_mask > 0] = colors[idx]
        
        # Create transparent overlay
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = leaf_mask
        
        # Get bounding rectangle with padding
        x, y, w, h = cv2.boundingRect(contour)
        padding = 10
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img.shape[1] - x, w + 2*padding), min(img.shape[0] - y, h + 2*padding)
        
        # Save individual leaf
        leaf_image = rgba[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_dir, f'leaf_{leaf_count+1}.png'), leaf_image)
        
        leaf_count += 1
    
    # Save visualization results
    cv2.imwrite(os.path.join(output_dir, '3_edge_detection.png'), edges)
    cv2.imwrite(os.path.join(output_dir, '4_contours.png'), contour_img)
    
    # Create final visualization with alpha blending
    alpha = 0.6
    blend = cv2.addWeighted(original, 1 - alpha, colored_masks, alpha, 0)
    cv2.imwrite(os.path.join(output_dir, '5_final_segmentation.png'), blend)
    cv2.imwrite(os.path.join(output_dir, '6_colored_masks.png'), colored_masks)
    
    return leaf_count

if __name__ == "__main__":
    input_image = "/home/nabin/Documents/DiseaseClassification/src/output_images/images_8090%_healthy_low/train/downey_mildew/2_3_30.png"
    output_directory = "leaf_segmentation"
    
    try:
        num_leaves = isolate_and_visualize_leaves(input_image, output_directory)
        print(f"Successfully processed {num_leaves} leaves. Check the '{output_directory}' folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")