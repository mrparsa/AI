import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import os

def convert_to_webp():
    file_paths = filedialog.askopenfilenames(title="Select Images", 
                                             filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_paths:
        return

    save_dir = filedialog.askdirectory(title="Select Save Directory")
    if not save_dir:
        return

    for file_path in file_paths:
        try:
            img = Image.open(file_path)
            base_name = os.path.basename(file_path)
            new_name = os.path.splitext(base_name)[0] + '.webp'
            save_path = os.path.join(save_dir, new_name)
            img.save(save_path, 'webp')
            img.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert {file_path}: {str(e)}")
            continue

    messagebox.showinfo("Conversion Complete", "All images were successfully converted to .webp format!")

root = tk.Tk()
root.title("Image Format Converter")
root.geometry("300x150")

convert_button = tk.Button(root, text="Convert Images to .webp", command=convert_to_webp)
convert_button.pack(pady=50)

root.mainloop()
