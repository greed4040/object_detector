import os
import random
import math
import cv2
import numpy as np
import json


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the new dimensions of the rotated image
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust the rotation matrix to take into account the translation
    rotation_matrix[0, 2] += (new_w - w) // 2
    rotation_matrix[1, 2] += (new_h - h) // 2
    
    # Create a 4-channel image with transparent background
    rotated_image = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    rotated_image[:, :, 3] = 0  # Set alpha channel to 0 for transparency
    
    # Rotate the image
    rotated_image[:, :, :3] = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderValue=(0, 0, 0, 0))
    
    # Create a mask for the rotated image
    mask = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, rotation_matrix, (new_w, new_h))
    rotated_image[:, :, 3] = mask
    
    return rotated_image

def place_image_on_a4(a4_image, input_image, x, y):
    h, w = input_image.shape[:2]
    
    # Extract the region of interest (ROI) from the A4 image
    roi = a4_image[y:y+h, x:x+w]
    
    # Create a mask based on the alpha channel of the input image
    mask = input_image[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply the mask to the ROI and input image
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    input_image_fg = cv2.bitwise_and(input_image[:, :, :3], input_image[:, :, :3], mask=mask)
    
    # Combine the masked ROI and input image
    combined = cv2.add(roi_bg, input_image_fg)
    
    # Replace the ROI in the A4 image with the combined image
    a4_image[y:y+h, x:x+w] = combined
    
    return a4_image, (x, y, x+w, y+h)

def generate_a4_image(input_dir, output_dir, num_images):
    a4_width, a4_height = 2480, 3508
    a4_image = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255
    
    input_images = []
    image_names = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            
            # Resize the image if it is larger than the A4 page
            h, w = image.shape[:2]
            if w > a4_width or h > a4_height:
                scale = min(a4_width / w, a4_height / h)
                new_width = int(w * scale)
                new_height = int(h * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            input_images.append(image)
            image_names.append(filename)
    
    if not input_images:
        print("No input images found.")
        return num_images  # Return the current num_images value
    
    num_classes = random.randint(1, min(4, len(input_images)))
    selected_indices = random.sample(range(len(input_images)), num_classes)
    selected_images = [input_images[i] for i in selected_indices]
    selected_names = [image_names[i] for i in selected_indices]
    
    bounding_boxes = []
    class_identifiers = {}
    for i, (image, name) in enumerate(zip(selected_images, selected_names), start=1):
        h, w = image.shape[:2]
        max_attempts = 100
        placed = False
        
        for _ in range(max_attempts):
            angle = random.uniform(-90, 90)
            rotated_image = rotate_image(image, angle)  # Rotate the image
            rotated_h, rotated_w = rotated_image.shape[:2]
            
            # Check if the rotated image fits within the A4 page dimensions
            if rotated_w > a4_width or rotated_h > a4_height:
                continue
            
            x = random.randint(0, a4_width - rotated_w)
            y = random.randint(0, a4_height - rotated_h)
            
            if x < 0.1 * rotated_w or x > a4_width - 0.9 * rotated_w or y < 0.1 * rotated_h or y > a4_height - 0.9 * rotated_h:
                continue
            
            overlap = False
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox["bbox"]
                if (max(x, x1) < min(x + rotated_w, x2) and max(y, y1) < min(y + rotated_h, y2)):
                    overlap_area = (min(x + rotated_w, x2) - max(x, x1)) * (min(y + rotated_h, y2) - max(y, y1))
                    if overlap_area > 0.1 * rotated_w * rotated_h:
                        overlap = True
                        break
            
            if not overlap:
                a4_image, bbox = place_image_on_a4(a4_image, rotated_image, x, y)  # Place the rotated image on the A4 page
                bounding_boxes.append({"bbox": list(bbox), "class": i})
                class_identifiers[i] = name
                placed = True
                break
    
    if bounding_boxes:
        output_filename = f"a4_image_{num_images}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, a4_image)
        
        bbox_filename = f"bounding_boxes_{num_images}.json"
        bbox_path = os.path.join(output_dir, bbox_filename)
        with open(bbox_path, "w") as f:
            json.dump(bounding_boxes, f)
        
        class_identifier_filename = f"class_identifiers_{num_images}.json"
        class_identifier_path = os.path.join(output_dir, class_identifier_filename)
        with open(class_identifier_path, "w") as f:
            json.dump(class_identifiers, f)
        
        num_images += 1
    
    return num_images

input_directory = "../downloads/cards/"
output_directory = "output"
num_generated_images = 0

num_images_to_generate = 50
while num_generated_images < num_images_to_generate:
    num_generated_images = generate_a4_image(input_directory, output_directory, num_generated_images)
