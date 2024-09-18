import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('test_images/12.png')

def watershed_segmentation(img, manual_thresh=100, adaptive_thresh=False):
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if adaptive_thresh:
        # Adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        # Manual thresholding (default uses Otsu if manual_thresh is not provided)
        if manual_thresh is None:
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            ret, thresh = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background (dilated regions)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Optional: Convert sure background to binary again
    sure_bg_binary = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY)[1]
    
    return sure_bg_binary

# Display the result
plt.imshow(watershed_segmentation(img), cmap='gray')
plt.show()
