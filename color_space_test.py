import numpy as np
import cv2
import skimage.io as io
import skimage.color as color
from skimage import transform
import matplotlib.pyplot as plt
import argparse

file_path = "D:/CSE Vision/CSE4310-main/assignments/assignment1/img/image.png"
img = cv2.imread(file_path)

def rgb_to_hsv_vectorized(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Calculate value (V) and chroma (C) using vectorized operations
    V = np.max(img, axis=2)
    C = V - np.min(img, axis=2)

    # Avoid division by zero
    C[C == 0] = 1  # Set C to 1 where it is 0 to avoid division by zero

    # Calculate saturation (S)
    S = C / V 

    # Calculate hue (H)
    Hue = np.zeros_like(V)
    
    # Perform element-wise comparisons
    condition1 = (C == 0)
    condition2 = (V == R)
    condition3 = (V == G)
    condition4 = (V == B)

    # Assign values based on conditions
    Hue[condition1] = 0
    Hue[condition2] = ((G - B) / C)[condition2] % 6
    Hue[condition3] = ((B - R) / C)[condition3] + 2
    Hue[condition4] = ((R - G) / C)[condition4] + 4
    
    #To convert into degrees 
    H = (Hue * 60) % 360 
    H = np.clip(H, 0, 360)

                # Scale Saturation and Value to [0, 1]
    S = np.clip(S, 0, 1)
    V = np.clip(V / 255.0, 0, 1)

                # Stack H, S, V to create the HSV image
    hsv_image = np.dstack((H, S, V))
    return hsv_image


def hsv_to_rgb_vectorized(img):
    H = img[:, :, 0]
    S = img[:, :, 1]
    V = img[:, :, 2] * 255.0  # Scale back to [0, 255]
    C = V * S
    Hue = H / 60
        
    X = C * (1 - np.abs(Hue % 2 - 1))
    
    condition1 = (Hue <= 0) & (Hue < 1)
    condition2 = (Hue <= 1) & (Hue < 2)
    condition3 = (Hue <= 2) & (Hue < 3)
    condition4 = (Hue <= 3) & (Hue < 4)
    condition5 = (Hue <= 4) & (Hue < 5)
    condition6 = (Hue <= 5) & (Hue < 6)
    
    R = np.zeros_like(Hue)
    G = np.zeros_like(Hue)
    B = np.zeros_like(Hue)
    
    R[condition1] = C[condition1]
    G[condition1] = X[condition1]
    B[condition1] = 0

    R[condition2] = X[condition2]
    G[condition2] = C[condition2]
    B[condition2] = 0

    R[condition3] = 0
    G[condition3] = C[condition3]
    B[condition3] = X[condition3]

    R[condition4] = 0
    G[condition4] = X[condition4]
    B[condition4] = C[condition4]

    R[condition5] = X[condition5]
    G[condition5] = 0
    B[condition5] = C[condition5]

    R[condition6] = C[condition6]
    G[condition6] = 0
    B[condition6] = X[condition6]
                
    m = V - C
    #Final RGB value can be calculated by adding the difference between the value and chroma to each pixel
    R_dash = R + m
    G_dash = G + m
    B_dash = B + m
                
    rgb_image = np.dstack((R_dash, G_dash, B_dash))
    return rgb_image
        
            
def main():
    hsv_image = rgb_to_hsv_vectorized(img)
    cv2.imshow("Original RGB Image", img)
    cv2.imshow("Converted HSV Image", (hsv_image * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    rgb_image = hsv_to_rgb_vectorized(hsv_image)
    cv2.imshow("HSV Image", (hsv_image * 255).astype(np.uint8))
    cv2.imshow("Converted RGB Image", (rgb_image).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()