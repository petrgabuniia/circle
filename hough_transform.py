import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread('test_images/12.png')
assert img is not None, "file could not be read, check with os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Otsu's binarization
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv.dilate(opening, kernel, iterations=3)
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
ret, markers = cv.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0


markers = cv.watershed(img, markers)


img[markers == -1] = [255, 0, 0]


plt.figure(figsize=(12, 12))
plt.subplot(231), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
plt.subplot(232), plt.imshow(thresh, cmap='gray'), plt.title('Thresholded Image')
plt.subplot(233), plt.imshow(sure_bg, cmap='gray'), plt.title('Sure Background')
plt.subplot(234), plt.imshow(dist_transform, cmap='gray'), plt.title('Distance Transform')
plt.subplot(235), plt.imshow(sure_fg, cmap='gray'), plt.title('Sure Foreground')
plt.subplot(236), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Segmented Image')

plt.show()
