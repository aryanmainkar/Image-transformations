import cv2

def resize_img(img, factor):
    # Check if the factor is valid
    if factor <= 0:
        print("Invalid scale factor")
        return None
    
    # Determine the new dimensions after resizing
    new_x = int(img.shape[0] * factor)
    new_y = int(img.shape[1] * factor)
    
    # Resize the image using nearest neighbor interpolation

    resized_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    return resized_img
    
    return resized_img

def create_img_pyramid(img, pyramid_height, filename):
    # Iterate over pyramid levels
    for level in range(1, pyramid_height + 1):
        # Calculate the scale factor for this level
        scale_factor = 2 ** level
        
        # Resize the image using the current scale factor
        resized_img = resize_img(img, scale_factor)
        
        # Save the resized image with the appropriate filename
        output_filename = f"{filename[:-4]}_{scale_factor}x.png"
        cv2.imwrite(output_filename, resized_img)
        print(f"Saved {output_filename}")
        

def main():
    file_path = "D:/CSE Vision/CSE4310-main/assignments/assignment1/img/image.png"
    img = cv2.imread(file_path)
    pyramid_height = 4
    create_img_pyramid(img, pyramid_height, 'image.png')
 
    
if __name__ == "__main__":
    main()
