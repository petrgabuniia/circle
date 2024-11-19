import tkinter as tk
from tkinter import Button, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime

# Global variables
show_circles = True
fig, ax, canvas = None, None, None
image, segmented_image, circles = None, None, None
min_precision = 0.05
sensor_name = "Sensor Name"
circle_text_handle = None


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

def refine_circle(image, circle, step_size=1, max_shift=5, max_radius_change=10):
    center = (int(circle[0]), int(circle[1]))
    radius = int(circle[2])
    best_center = center
    best_radius = radius
    best_precision = calculate_precision(image, center, radius)
    for dx in range(-max_shift, max_shift + 1, step_size):
        for dy in range(-max_shift, max_shift + 1, step_size):
            new_center = (center[0] + dx, center[1] + dy)
            new_precision = calculate_precision(image, new_center, radius)
            if new_precision < best_precision:
                best_center = new_center
                best_precision = new_precision
    for i in range(1, max_radius_change + 1):
        new_radius = radius - i
        if new_radius <= 0:
            break
        precision = calculate_precision(image, best_center, new_radius)
        if precision < best_precision:
            best_radius = new_radius
            best_precision = precision
    for i in range(1, max_radius_change + 1):
        new_radius = radius + i
        precision = calculate_precision(image, best_center, new_radius)
        if precision <= best_precision:
            best_radius = new_radius
            best_precision = precision
    return (best_center[0], best_center[1], best_radius)

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
                refined_circle = refine_circle(image, i)
                valid_circles.append(refined_circle)
            #print(f"Circle at ({i[0]}, {i[1]}) with radius {i[2]} has precision: {precision:.2f}%")
        if len(valid_circles) > 0:
            valid_circles = np.expand_dims(np.array(valid_circles), axis=0)
            print(f"Number of valid circles detected: {len(valid_circles[0])}")
            valid_circles = remove_intersecting_circles(valid_circles)
            print(f"Number of circles after removing intersections: {len(valid_circles[0])}")
            for circle in valid_circles[0]:
                print(f"Circle at ({circle[0]}, {circle[1]}) with radius {circle[2]}")
            return valid_circles
            
        else:
            print("No valid circles detected")
            return None
    else:
        print("No circles detected")
        return None



def watershed_segmentation(img, manual_thresh, adaptive_thresh):
    """
    Performs watershed segmentation on an image and returns the segmented image.
    """
    assert img is not None, "File could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if adaptive_thresh:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        if manual_thresh is None:
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            ret, thresh = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    sure_bg_bgr = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)

    return sure_bg_bgr

def draw_circles(circles, image, color=(0, 255, 0)):
    """
    Draws circles on the given image. The default color is green, but you can pass in another color.
    """
    global show_circles

    if circles is None:
        return

    image_with_circles = image.copy()
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        if show_circles:
            cv2.circle(image_with_circles, center, radius, color=color, thickness=1)
            cv2.circle(image_with_circles, center, radius=10, color=(0, 0, 255), thickness=-1)

    return image_with_circles

    

def toggle_circles():
    global show_circles
    show_circles = not show_circles

    if circles is not None:
        xlim_original = ax_original.get_xlim()
        ylim_original = ax_original.get_ylim()
        xlim_segmented = ax_segmented.get_xlim()
        ylim_segmented = ax_segmented.get_ylim()

        img_with_circles = draw_circles(circles, image)
        ax_original.clear()
        ax_original.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
        ax_original.set_xlim(xlim_original)
        ax_original.set_ylim(ylim_original)
        ax_original.set_title("Original Image with Circles")

        ax_tosave.clear()
        ax_tosave.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
        ax_tosave.set_xlim(xlim_original)
        ax_tosave.set_ylim(ylim_original)
        ax_tosave.set_title("Original Image with Circles")

        img_with_circles_segmented = draw_circles(circles, segmented_image, color=(0, 0, 255))
        ax_segmented.clear()
        ax_segmented.imshow(cv2.cvtColor(img_with_circles_segmented, cv2.COLOR_BGR2RGB))
        ax_segmented.set_xlim(xlim_segmented)
        ax_segmented.set_ylim(ylim_segmented)
        ax_segmented.set_title("Watershed Image with Circles (in color)")

        canvas.draw()

def create_matplotlib_window():
    global fig_to_show, fig_to_save, ax_original, ax_segmented, ax_tosave, canvas, frame

    fig_to_show = Figure(figsize=(10, 6))
    fig_to_save = Figure(figsize=(10, 6))
    ax_original = fig_to_show.add_subplot(121)
    ax_segmented = fig_to_show.add_subplot(122)
    ax_tosave = fig_to_save.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig_to_show, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_to_show.subplots_adjust(bottom=0.4)
    fig_to_save.subplots_adjust(bottom=0.4)
    update_circle_text("No circles detected", current_time)  # Initialize with no circles

def update_circle_text(circle_details, current_time):
    global circle_text_handle, fig_to_show, fig_to_save
    if circle_text_handle is not None:
        circle_text_handle.remove()  # Remove the previous text to avoid overlap
    circle_text_handle = fig_to_show.text(0.5, 0.1, circle_details, ha='center', fontsize=10, transform=fig_to_show.transFigure)
    fig_to_show.text(0.5, 0.05, f"Generated on: {current_time}", ha='center', fontsize=10, style='italic', transform=fig_to_show.transFigure)
    fig_to_save.text(0.5, 0.1, circle_details, ha='center', fontsize=10, transform=fig_to_save.transFigure)
    fig_to_save.text(0.5, 0.05, f"Generated on: {current_time}", ha='center', fontsize=10, style='italic', transform=fig_to_save.transFigure)
    canvas.draw()

def display_images():
    global canvas, ax_original, ax_segmented, circles

    if circles is not None and circles.any():
        img_with_circles = draw_circles(circles, image)
        img_with_circles_segmented = draw_circles(circles, segmented_image, color=(0, 255, 0))
        circle_details = "\n".join([f"Circle at ({circle[0]}, {circle[1]}) with radius {circle[2]}" for circle in circles[0]])
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_circle_text(circle_details, current_time)
    else:
        update_circle_text("No circles detected", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    ax_original.clear()
    ax_segmented.clear()
    ax_original.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
    ax_segmented.imshow(cv2.cvtColor(img_with_circles_segmented, cv2.COLOR_BGR2RGB))
    ax_original.set_title("Original Image with Circles")
    ax_segmented.set_title("Watershed Image with Circles (in color)")

    canvas.draw()


def load_image():
    global image, segmented_image, circles, sensor_name

    file_path = filedialog.askopenfilename()
    if file_path:
        sensor_name = file_path.split("/")[-1].split(".")[0]
        image = cv2.imread(file_path)
        segmented_image = watershed_segmentation(image, manual_thresh=60, adaptive_thresh=False)
        circles = find_circles(segmented_image)
        display_images()


def generate_report():
    global circles, fig_to_save

    if circles is not None:
        img_tosave = draw_circles(circles, image)
        ax_tosave.clear()
        ax_tosave.imshow(cv2.cvtColor(img_tosave, cv2.COLOR_BGR2RGB))
        ax_tosave.set_title("Original Image with Circles")

        save_path = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[('PDF files', '*.pdf')])
        if save_path:  # Check if a path was chosen
            fig_to_save.savefig(save_path, format='pdf', dpi=1000)
            messagebox.showinfo("Save Report", f"Report saved as {save_path}")
        else:
            messagebox.showinfo("Save Report", "Saving cancelled.")
    else:
        messagebox.showinfo("Save Report", "No image loaded or no circles detected to generate a report.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Circle Detection Tool")

    frame = tk.Frame(root)
    frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    create_matplotlib_window()

    load_button = Button(root, text="Load Image", command=load_image)
    load_button.pack(side=tk.BOTTOM)

    save_button = Button(root, text="Generate Report", command=generate_report)
    save_button.pack(side=tk.BOTTOM)

    toggle_button = Button(root, text="Show/Hide", command=toggle_circles)
    toggle_button.pack(side=tk.BOTTOM)

    root.mainloop()