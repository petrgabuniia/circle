import cv2
import numpy as np

def find_circles(image):
    if image is None:
        print("Failed to load image")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=2000)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Number of circles detected: {len(circles[0, :])}")
        return circles
    else:
        print("No circles detected")
        return None
    

def calculate_precision(image, center, radius):
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image_rgb.shape
    radius = int(radius)
    center_x, center_y = center
    center_color = image_rgb[center_y, center_x]

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circle_mask = dist_from_center <= radius

    circle_pixels = image_rgb[circle_mask]

    same_color_count = np.sum(np.all(circle_pixels == center_color, axis=1))
    total_circle_pixels = np.sum(circle_mask)

    if total_circle_pixels == 0:
        return 0.0
    
    percentage = (same_color_count / total_circle_pixels) * 100
    return percentage     


def draw_circles(circles, image):
    if circles is None:
        return
    
    image_with_circles = image.copy()

    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        print("center : ", center, "radius: ", radius)
        
        cv2.circle(image_with_circles, center, radius=10, color=(0, 0, 255), thickness=-1)
        cv2.circle(image_with_circles, center, radius, color=(0, 255, 0), thickness=3)
        
        cv2.imshow("Circles Detected", image_with_circles)
        cv2.waitKey(0) 
        
        center_image = np.zeros_like(image_with_circles)
        cv2.circle(center_image, center, radius=10, color=(0, 0, 255), thickness=-1)
        
        precision = calculate_precision(image_with_circles, center, radius)
        print(f'Approximation Precision: {precision:.2f}%')
    
    cv2.destroyAllWindows()


image = cv2.imread('test_images/3.png', cv2.IMREAD_COLOR)


circles = find_circles(image)

draw_circles(circles, image)
