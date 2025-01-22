import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance

# Set up input and output directories
DATA_DIR = './data'
AUGMENTED_DATA_DIR = './augmented_data'

# Create output directory if it doesn't exist
if not os.path.exists(AUGMENTED_DATA_DIR):
    os.makedirs(AUGMENTED_DATA_DIR)

# Define augmentation functions
def rotate_image(image, angle):
    """Rotate image by a specific angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def flip_image(image):
    """Flip image horizontally."""
    return cv2.flip(image, 1)

def resize_image(image, scale=1.2):
    """Resize image by a specific scale."""
    (h, w) = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size)

def add_noise(image):
    """Add random noise to the image."""
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

def adjust_brightness(image, factor=1.5):
    """Adjust brightness of the image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

# Process each folder (each alphabet)
for dir_name in os.listdir(DATA_DIR):
    input_dir = os.path.join(DATA_DIR, dir_name)
    output_dir = os.path.join(AUGMENTED_DATA_DIR, dir_name)
    
    # Create output directory for each alphabet if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the folder
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        # Skip if the image could not be read
        if image is None:
            continue
        
        # Apply augmentations
        augmented_images = []
        augmented_images.append(rotate_image(image, angle=random.randint(-30, 30)))  # Random rotation
        augmented_images.append(flip_image(image))  # Horizontal flip
        augmented_images.append(resize_image(image, scale=random.uniform(0.8, 1.2)))  # Random resize
        augmented_images.append(add_noise(image))  # Add noise
        augmented_images.append(adjust_brightness(image, factor=random.uniform(0.5, 1.5)))  # Adjust brightness

        # Save augmented images
        for i, aug_img in enumerate(augmented_images):
            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.png"
            cv2.imwrite(os.path.join(output_dir, aug_img_name), aug_img)

print("Image augmentation completed and saved to 'augmented_data' folder.")
