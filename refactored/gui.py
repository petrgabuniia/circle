import tkinter as tk
from tkinter import Button, filedialog, messagebox, Entry, Label, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2
from datetime import datetime
import image_processing as ip
import config_manager as cm

# Module-level variables
show_circles = True
fig_to_show = None
fig_to_save = None
ax_original = None
ax_segmented = None
ax_tosave = None
canvas = None
image = None
segmented_image = None
circles = None
min_precision = 0.05
circle_text_handle = []
general_text_handle = []
operator_name_var = None
standard_name_var = None
standard_id_var = None
unit_id_var = None

def setup_gui(root):
    global operator_name_var, standard_name_var, standard_id_var, unit_id_var
    config_values = cm.load_config()
    operator_name_var = tk.StringVar(value=config_values['operator_name'])
    standard_name_var = tk.StringVar(value=config_values['standard_name'])
    standard_id_var = tk.StringVar(value=config_values['standard_id'])
    unit_id_var = tk.StringVar(value=config_values['unit_id'])

    right_frame = tk.Frame(root)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    add_info_fields(right_frame)
    add_manual_thresh_field(right_frame)
    create_matplotlib_window(left_frame)

def add_info_fields(main_frame):
    frame = tk.Frame(main_frame)
    frame.pack(fill=tk.BOTH, expand=True)

    operator_entry = Entry(frame, textvariable=operator_name_var)
    operator_entry.grid(row=0, column=1, sticky='ew')
    Label(frame, text="Operator Name:").grid(row=0, column=0, sticky='w')

    standard_entry = Entry(frame, textvariable=standard_name_var)
    standard_entry.grid(row=1, column=1, sticky='ew')
    Label(frame, text="Standard Name:").grid(row=1, column=0, sticky='w')

    standard_id_entry = Entry(frame, textvariable=standard_id_var)
    standard_id_entry.grid(row=2, column=1, sticky='ew')
    Label(frame, text="Standard ID:").grid(row=2, column=0, sticky='w')

    unit_id_entry = Entry(frame, textvariable=unit_id_var)
    unit_id_entry.grid(row=3, column=1, sticky='ew')
    Label(frame, text="Unit ID:").grid(row=3, column=0, sticky='w')

    update_button = Button(frame, text="Update Info", command=update_info)
    update_button.grid(row=4, column=0, columnspan=2)

    load_button = tk.Button(frame, text="Load Image", command=load_image)
    load_button.grid(row=5, column=0, columnspan=2)

    save_button = tk.Button(frame, text="Generate Report", command=generate_report)
    save_button.grid(row=6, column=0, columnspan=2)

    toggle_button = tk.Button(frame, text="Show/Hide Circles", command=toggle_circles)
    toggle_button.grid(row=7, column=0, columnspan=2)

def add_manual_thresh_field(main_frame):
    global manual_thresh_var
    manual_thresh_var = tk.StringVar(value="60")

    frame = tk.Frame(main_frame)
    frame.pack(fill=tk.BOTH, expand=True, pady=10)

    manual_thresh_entry = Entry(frame, textvariable=manual_thresh_var)
    manual_thresh_entry.grid(row=0, column=1, sticky='ew')
    Label(frame, text="Manual Threshold:").grid(row=0, column=0, sticky='w')

    recalc_button = Button(frame, text="Recalculate", command=lambda: recalculate_image(int(manual_thresh_var.get())))
    recalc_button.grid(row=1, column=1)

    question_btn = Button(frame, text="?", command=show_threshold_info)
    question_btn.grid(row=0, column=2, padx=5)

def show_threshold_info():
    messagebox.showinfo("Threshold Information", "Threshold is a value used to determine the boundary level for pixel values in the image processing. Lower threshold values will decrease the number of found circles and their radius while increasing it will result opposite.")

def recalculate_image(new_thresh):
    global image, segmented_image, circles
    if image is not None:
        segmented_image = ip.watershed_segmentation(image, manual_thresh=new_thresh, adaptive_thresh=False)
        circles = ip.find_circles(segmented_image, min_precision)
        display_images()
    else:
        messagebox.showerror("Recalculate", "No image loaded to recalculate.")

def update_info():
    cm.save_config(
        operator_name_var.get(),
        standard_name_var.get(),
        standard_id_var.get(),
        unit_id_var.get()
    )
    messagebox.showinfo("Update Info", "Information updated successfully.")

def create_matplotlib_window(right_frame):
    global fig_to_show, fig_to_save, ax_original, ax_segmented, ax_tosave, canvas
    fig_to_show = Figure(figsize=(10, 6))
    fig_to_save = Figure(figsize=(10, 6))
    ax_original = fig_to_show.add_subplot(121)
    ax_segmented = fig_to_show.add_subplot(122)
    ax_tosave = fig_to_save.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig_to_show, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, right_frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.mpl_connect('draw_event', on_draw_event)

def on_draw_event(event):
    if ax_original.get_images():
        sync_figures_for_save()
        canvas.draw_idle()

def sync_figures_for_save():
    if image is not None:
        img_to_save = ip.draw_circles(circles, image, show_circles)
        ax_tosave.clear()
        ax_tosave.imshow(cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB))
        ax_tosave.set_title("Report")

def display_images():
    global canvas, ax_original, ax_segmented, ax_tosave
    if image is not None:
        img_with_circles = ip.draw_circles(circles, image, show_circles)
        img_with_circles_segmented = ip.draw_circles(circles, segmented_image, show_circles, color=(0, 0, 255))
        circle_details = "\n".join([f"Circle ({circle[0]}, {circle[1]}) radius {circle[2]}" for circle in circles[0]]) if circles is not None else "No circles detected"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_circle_text(circle_details)
        update_general_info(current_time)
        ax_original.clear()
        ax_original.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
        ax_original.set_title("Original Image with Circles")
        ax_segmented.clear()
        ax_segmented.imshow(cv2.cvtColor(img_with_circles_segmented, cv2.COLOR_BGR2RGB))
        ax_segmented.set_title("Watershed Image with Circles (in color)")
        canvas.draw()
        sync_figures_for_save()

def update_circle_text(circle_details):
    global circle_text_handle, fig_to_show, fig_to_save
    if circle_text_handle:
        for text in circle_text_handle:
            text.remove()
    circle_text_handle = [
        fig_to_show.text(0.5, 0.01, circle_details, ha='center', fontsize=10, transform=fig_to_show.transFigure),
        fig_to_save.text(0.01, 0.5, circle_details, ha='left', va='center', fontsize=10, transform=fig_to_save.transFigure)
    ]
    canvas.draw()

def update_general_info(current_time):
    global general_text_handle, fig_to_show, fig_to_save
    if general_text_handle:
        for text in general_text_handle:
            text.remove()
    general_text_handle = [
        fig_to_show.text(0.5, 0.9, f"Generated on: {current_time}", ha='center', fontsize=10, style='italic', transform=fig_to_show.transFigure),
        fig_to_save.text(0.01, 0.1, f"Generated on: {current_time}", ha='left', fontsize=10, style='italic', transform=fig_to_save.transFigure)
    ]
    canvas.draw()

def toggle_circles():
    global show_circles
    show_circles = not show_circles
    display_images()

def load_image():
    global image, segmented_image, circles
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            print("Image loaded successfully.")
            segmented_image = ip.watershed_segmentation(image, manual_thresh=60, adaptive_thresh=False)
            circles = ip.find_circles(segmented_image, min_precision)
            display_images()
        else:
            messagebox.showerror("Load Image", "Failed to load the image.")
            print("Failed to load image at:", file_path)

def generate_report():
    global fig_to_save, canvas
    if circles is not None:
        params_text = f"Operator: {operator_name_var.get()}\nStandard ID: {standard_name_var.get()}\nUnit ID: {standard_id_var.get()}\nUnit: {unit_id_var.get()}"
        info_text_handle = fig_to_save.text(0.01, 0.25, params_text, ha='left', va='top', fontsize=10, transform=fig_to_save.transFigure)
        dpi_input = simpledialog.askinteger("Input DPI", "Enter the DPI for the report:", minvalue=72, maxvalue=1200, initialvalue=300)
        if dpi_input is None:
            messagebox.showinfo("Save Report", "Report generation cancelled.")
            info_text_handle.remove()
            canvas.draw()
            return
        current_time = datetime.now().strftime("%Y-%m-%d %H-%M")
        unit_id = unit_id_var.get()
        default_filename = f"Resolution Report {current_time} ID {unit_id}.pdf"
        save_path = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[('PDF files', '*.pdf')], initialfile=default_filename)
        if save_path:
            fig_to_save.savefig(save_path, format='pdf', dpi=dpi_input)
            messagebox.showinfo("Save Report", f"Report saved as {save_path}")
            info_text_handle.remove()
            canvas.draw()
        else:
            messagebox.showinfo("Save Report", "Saving cancelled.")
            info_text_handle.remove()
            canvas.draw()
    else:
        messagebox.showinfo("Save Report", "No image loaded or no circles detected to generate a report.")