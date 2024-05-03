import os
import random
import math
import cv2
import numpy as np

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(255, 255, 255))
    return rotated_image

def place_image_on_a4(a4_image, input_image, x, y, angle):
    h, w = input_image.shape[:2]
    rotated_image = rotate_image(input_image, angle)
    a4_image[y:y+h, x:x+w] = rotated_image
    return a4_image, (x, y, x+w, y+h)

def generate_a4_image(input_dir, output_dir, num_images):
    a4_width, a4_height = 2480, 3508
    a4_image = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255
    
    input_images = []
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        input_images.append(image)
    
    num_classes = random.randint(1, min(4, len(input_images)))
    selected_images = random.sample(input_images, num_classes)
    
    bounding_boxes = []
    for image in selected_images:
        h, w = image.shape[:2]
        max_attempts = 100
        placed = False
        
        for _ in range(max_attempts):
            angle = random.uniform(-10, 10)
            x = random.randint(0, a4_width - w)
            y = random.randint(0, a4_height - h)
            
            if x < 0.1 * w or x > a4_width - 0.9 * w or y < 0.1 * h or y > a4_height - 0.9 * h:
                continue
            
            overlap = False
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox
                if (max(x, x1) < min(x + w, x2) and max(y, y1) < min(y + h, y2)):
                    overlap_area = (min(x + w, x2) - max(x, x1)) * (min(y + h, y2) - max(y, y1))
                    if overlap_area > 0.1 * w * h:
                        overlap = True
                        break
            
            if not overlap:
                a4_image, bbox = place_image_on_a4(a4_image, image, x, y, angle)
                bounding_boxes.append(bbox)
                placed = True
                break
        
        if not placed:
            print("Failed to place an image without overlapping.")
    
    output_filename = f"a4_image_{num_images}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, a4_image)
    
    bbox_filename = f"bounding_boxes_{num_images}.txt"
    bbox_path = os.path.join(output_dir, bbox_filename)
    with open(bbox_path, "w") as f:
        for bbox in bounding_boxes:
            f.write(",".join(map(str, bbox)) + "\n")
    
    num_images += 1
    return num_images

input_directory = "input_images"
output_directory = "output_images"
num_generated_images = 0

num_images_to_generate = 100
while num_generated_images < num_images_to_generate:
    num_generated_images = generate_a4_image(input_directory, output_directory, num_generated_images)
