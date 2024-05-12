import numpy as np
import cv2 
import sys
from random import randint
from numpy.lib import stride_tricks

def random_crop(img, size):
    
    height, width, _ = img.shape
    loaded_img_size = height * width
    
    if len(img.shape) != 3:
        print("Input image must be a 3D array")
        return None

    if size <= 0 or size <= loaded_img_size :
        x = randint(0, width - size + 1)
        y = randint(0, height - size + 1)
        return img[y:y+size, x:x+size, :]
    

def extract_patches(img, num_patches):
    # Get the dimensions of the input image
    height, width, channels = img.shape
    
    # Calculate the size of each patch
    patch_size = height // num_patches
    
    # Reshape the image into a 4D array where the first two dimensions represent
    # the number of patches along each axis, and the last two dimensions represent
    # the size of each patch
    patches = img.reshape(num_patches, patch_size, num_patches, patch_size, channels)
    
    # Swap the axes to rearrange the patches into a more intuitive order
    patches = np.swapaxes(patches, 1, 2)
    
    # Flatten the patches along the first two dimensions to get a list of patches
    patches = patches.reshape(-1, patch_size, patch_size, channels)
    
    return patches   

def resize_img(img, factor):
    if factor <= 0:
        print("Invalid scale input")
        return None
    
    height = int(img.shape[0] * factor)
    width = int(img.shape[1] * factor)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return resized_img

def color_jitter(img, hue, saturation, value):
    # Convert the input image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Generate random perturbations for each channel
    hue_delta = np.random.randint(-hue, hue + 1)
    saturation_delta = np.random.randint(-saturation, saturation + 1)
    value_delta = np.random.randint(-value, value + 1)
    print(f'Hue Delta: {hue_delta}, Saturation Delta: {saturation_delta}, Value Delta: {value_delta}')
    # Apply the perturbations to the HSV channels
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_delta) % 180  # Hue
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + saturation_delta, 0, 255)  # Saturation
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + value_delta, 0, 255)  # Value
    
    # Convert the modified HSV image back to BGR color space
    jittered_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    return jittered_img
    

def main():   
    file_path = "D:/CSE Vision/CSE4310-main/assignments/assignment1/img/image.png"
    img = cv2.imread(file_path)
    
    resized_img = resize_img(img, 0.5)
    cv2.imshow('Original Image', img)
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)
    size = 100  
    
    jittered_img = color_jitter(img, 30, 30, 30)
    cv2.imshow('Original Image', img)
    cv2.imshow('Jittered Image', jittered_img)
    cv2.waitKey(0)
    
    if size <= min(img.shape[:2]):
        cropped_image = random_crop(img, size)
        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
    else:
        print("Crop size > Inputted Image Size \nExiting..")
        sys.exit()


    
    
if __name__ == "__main__":
    main()
