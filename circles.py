import cv2
import numpy as np

def find_circles(image):
    if image is None:
        print("Failed to load image")
        return None
    
    # Check if the image has only one channel (grayscale or binary)
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=10, minRadius=5, maxRadius=2000)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        valid_circles = []

        for i in circles[0, :]:
            precision = calculate_precision(image, (i[0], i[1]), i[2])
            if precision < min_precision:
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

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to get a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Perform morphological opening to remove noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate to obtain sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Ensure output is binary: 0 or 255
    sure_bg_binary = cv2.threshold(sure_bg, 127, 255, cv2.THRESH_BINARY)[1]

    return sure_bg_binary


   

def calculate_precision(image, center, radius):
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert the image to grayscale as it contains only black and white pixels
    if len(image.shape) == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image_gray.shape
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return 0
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    circle_mask = dist_from_center <= radius

    circle_pixels = image_gray[circle_mask]

    # Count black and white pixels (black pixels have value 0 in grayscale)
    black_pixel_count = np.sum(circle_pixels == 0)
    total_circle_pixels = len(circle_pixels)

    if total_circle_pixels == 0:
        return 0.0
    
    # Ratio of black pixels to all pixels inside the circle
    ratio = black_pixel_count / total_circle_pixels
    return ratio



def draw_circles(circles, image):
    if circles is None:
        return
    
    image_with_circles = image.copy()
    image_tranformed = watershed_segmentation(image)


    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        print("center : ", center, "radius: ", radius)
        
        # Draw the circle's center and perimeter
        cv2.circle(image_with_circles, center, radius=10, color=(0, 0, 255), thickness=-1)
        cv2.circle(image_with_circles, center, radius, color=(0, 255, 0), thickness=3)
        
    
    # Display the image with all circles
    cv2.imshow("Transformed Image", image_tranformed)
    cv2.imshow("Circles Detected", image_with_circles)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

min_precision = 0.05
image = cv2.imread('test_images/7.png', cv2.IMREAD_COLOR)

circles = find_circles(watershed_segmentation(image))

draw_circles(circles, image)
