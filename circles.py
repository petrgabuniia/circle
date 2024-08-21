import cv2
import numpy as np

def find_circle_diameter(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print("Failed to load image")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=65, minRadius=700, maxRadius=2000)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        image_with_circles = image.copy()
        print (f"Number of circles detected: {len(circles[0, :])}")
        
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            print(center, radius)
            
            cv2.circle(image_with_circles, center, radius=10, color=(0, 0, 255), thickness=-1)
            cv2.circle(image_with_circles, center, radius, color=(0, 255, 0), thickness=10)
            
            precision = calculate_precision(image, center, 2*radius)
            print(f'Approximation Precision: {precision:.2f}%')

       

    else:
        print("No circles detected")
    
def calculate_precision(image, center, radius):
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image_rgb.shape
    radius = int(radius)
    center_color = image_rgb[center[0], center[1]]

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    full_circle_mask = dist_from_center <= radius
    clipped_mask = np.zeros_like(full_circle_mask, dtype=bool)
    clipped_mask[np.where(full_circle_mask)] = True
    clipped_mask[dist_from_center > radius] = False
    circle_pixels = image_rgb[clipped_mask]
    same_color_count = np.sum(np.all(circle_pixels == center_color, axis=1))
    total_circle_pixels = np.sum(clipped_mask)

    if total_circle_pixels == 0:
        return 0.0 

    percentage = (same_color_count / total_circle_pixels) * 100
    return percentage

find_circle_diameter('circle.png')


