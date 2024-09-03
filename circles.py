import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_intersection_area(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    
    x1, y1, r1 = float(x1), float(y1), float(r1)
    x2, y2, r2 = float(x2), float(y2), float(r2)
    
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if d >= r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    
    r1_squared = r1 ** 2
    r2_squared = r2 ** 2

    part1 = r1_squared * np.arccos((d**2 + r1_squared - r2_squared) / (2 * d * r1))
    part2 = r2_squared * np.arccos((d**2 + r2_squared - r1_squared) / (2 * d * r2))
    part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    
    intersection_area = part1 + part2 - part3
    return intersection_area

def is_completely_inside(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    
    x1, y1, r1 = float(x1), float(y1), float(r1)
    x2, y2, r2 = float(x2), float(y2), float(r2)
    
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance + r1 <= r2

def should_remove(circle1, circle2):
    area1 = np.pi * circle1[2] ** 2
    area2 = np.pi * circle2[2] ** 2

    intersection_area = calculate_intersection_area(circle1, circle2)

    if is_completely_inside(circle1, circle2):
        return circle1
    if is_completely_inside(circle2, circle1):
        return circle2

    if intersection_area > 0.5 * min(area1, area2):
        return circle1 if area1 < area2 else circle2

    return None

def remove_intersecting_circles(circles):
    if circles is None or len(circles) == 0:
        return circles
    
    circles = [tuple(c) for c in circles[0]]
    while True:
        to_remove = set()
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                circle1 = circles[i]
                circle2 = circles[j]
                removal_candidate = should_remove(circle1, circle2)
                if removal_candidate is not None:
                    to_remove.add(removal_candidate)
        
        if not to_remove:
            break
        
        circles = [c for c in circles if c not in to_remove]
    
    if len(circles) == 0:
        print("All circles were removed, reverting to original set")
        circles = [tuple(c) for c in circles[0]]
    
    return np.expand_dims(np.array(circles), axis=0)

def find_circles(image):
    if image is None:
        print("Failed to load image")
        return None
    
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=75, param1=20, param2=15, minRadius=10, maxRadius=2000)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        valid_circles = []

        for i in circles[0, :]:
            precision = calculate_precision(image, (i[0], i[1]), i[2])
            if precision < min_precision:
                valid_circles.append(i)
            print(f"Circle at ({i[0]}, {i[1]}) with radius {i[2]} has precision: {precision:.2f}%")
        
        if len(valid_circles) > 0:
            valid_circles = np.expand_dims(np.array(valid_circles), axis=0)
            print(f"Number of valid circles detected: {len(valid_circles[0])}")
            
            # Remove intersecting circles here
            valid_circles = remove_intersecting_circles(valid_circles)
            print(f"Number of circles after removing intersections: {len(valid_circles[0])}")
            
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
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    sure_bg_binary = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY)[1]

    return sure_bg_binary

def calculate_precision(image, center, radius):
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    if len(image.shape) == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image_gray.shape
    center_x, center_y = int(center[0]), int(center[1])
    radius = int(radius)

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    circle_mask = dist_from_center <= radius

    valid_mask = (X - center_x >= -radius) & (X - center_x <= radius) & \
                 (Y - center_y >= -radius) & (Y - center_y <= radius)

    combined_mask = circle_mask & valid_mask

    circle_pixels = image_gray[combined_mask]

    black_pixel_count = np.sum(circle_pixels == 0)
    total_circle_pixels = np.sum(circle_mask)

    total_circle_pixels = total_circle_pixels if total_circle_pixels > 0 else 1

    ratio = black_pixel_count / total_circle_pixels
    return ratio


def draw_circles(circles, image):
    if circles is None:
        return
    
    image_with_circles = image.copy()
    image_transformed = watershed_segmentation(image)

    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        print("center : ", center, "radius: ", radius)
        
        cv2.circle(image_with_circles, center, radius=10, color=(0, 0, 255), thickness=-1)
        cv2.circle(image_with_circles, center, radius, color=(0, 255, 0), thickness=3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_circles_rgb = cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB)
    image_transformed_rgb = cv2.cvtColor(image_transformed, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(image_transformed_rgb)
    plt.title("Transformed Image")

    plt.subplot(1, 3, 3)
    plt.imshow(image_with_circles_rgb)
    plt.title("Circles Detected")

    plt.show()

min_precision = 0.05
image = cv2.imread('test_images/10.png', cv2.IMREAD_COLOR)

circles = find_circles(watershed_segmentation(image))

draw_circles(circles, image)
