import cv2
import numpy as np

def find_circles(image):
    if image is None:
        print("Failed to load image")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=25, minRadius=10, maxRadius=2000)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        valid_circles = []

        for i in circles[0, :]:
            precision = calculate_precision(image, (i[0], i[1]), i[2])
            if precision < 10:
                valid_circles.append(i)
            print(f"Circle at ({i[0]}, {i[1]}) with radius {i[2]} has precision: {precision:.2f}%")
        
        valid_circles = np.array(valid_circles, dtype=np.uint16)
        if len(valid_circles) > 0:
            valid_circles = np.expand_dims(valid_circles, axis=0)  
            print(f"Number of valid circles detected: {len(valid_circles[0, :])}")
            return valid_circles
        else:
            print("No valid circles detected")
            return None
    else:
        print("No circles detected")
        return None
        

def watershed_segmentation(img):
    assert img is not None, "file could not be read, check with os.path.exists()"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    sure_bg_color = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)

    return sure_bg_color

   

def calculate_precision(image, center, radius, color_threshold=20):
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image_rgb.shape
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return 0
    center = (int(center[0]), int(center[1]))
    radius = int(radius)
    center_color = image_rgb[center[1], center[0]]

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    circle_mask = dist_from_center <= radius

    circle_pixels = image_rgb[circle_mask]

    # Calculate the Euclidean distance in RGB space
    color_distances = np.sqrt(np.sum((circle_pixels - center_color)**2, axis=1))

    # Count pixels where the color distance is greater than the specified threshold
    different_color_count = np.sum(color_distances > color_threshold)
    total_circle_pixels = np.sum(circle_mask)

    if total_circle_pixels == 0:
        return 0.0
    
    percentage = (different_color_count / total_circle_pixels) * 100
    return percentage



def draw_circles(circles, image):
    if circles is None:
        return
    
    image_with_circles = image.copy()

    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        print("center : ", center, "radius: ", radius)
        
        # Draw the circle's center and perimeter
        cv2.circle(image_with_circles, center, radius=10, color=(0, 0, 255), thickness=-1)
        cv2.circle(image_with_circles, center, radius, color=(0, 255, 0), thickness=3)
        
    
    # Display the image with all circles
    cv2.imshow("Circles Detected", image_with_circles)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


image = cv2.imread('test_images/13.png', cv2.IMREAD_COLOR)

circles = find_circles(watershed_segmentation(image))

draw_circles(circles, image)
