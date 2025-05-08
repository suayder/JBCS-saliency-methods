"""

"""
import os
import argparse
from PIL import Image

def create_horizontal_strip(input_folder, output_file, n, frame_ratio):
    # Get all image files in the folder
    image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    # TODO: REMOVER
    # image_files = [image_files[i] for i in [1544, 2408, 2450, 2527, 2702, 3065, 3175, 3640, 4053]]
    image_files = [os.path.join(input_folder, i) for i in ['00025.jpg','00260.jpg', '00741.jpg', '01933.jpg']]
    
    if len(image_files) < n:
        raise ValueError("Not enough images in the folder to create the strip with the specified number of images.")

    # Select images based on frame ratio
    selected_images = image_files[::frame_ratio][:n]

    # Open images and calculate total width and max height
    images = [Image.open(img) for img in selected_images]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image
    strip = Image.new('RGB', (total_width, max_height))

    # Paste images side by side
    x_offset = 0
    y_offset = 0
    for img in images:
        strip.paste(img, (x_offset, 0))
        x_offset += img.size[0]
        y_offset += img.size[1]

    # Save the resulting image
    strip.save(output_file)
    print(f"Horizontal strip saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a horizontal image strip from frames in a folder.")
    parser.add_argument("--input_folder", type=str, default="/data/JBCS_paper/crop/method2/tmfi-croped-centered/", help="Path to the folder containing the images. Default is the current directory.")
    parser.add_argument("--output_file", type=str, default="tac_crop.jpg", help="Path to save the resulting horizontal strip. Default is 'output_strip.jpg'.")
    parser.add_argument("--n", type=int, default=4, help="Number of images to include in the strip. Default is 5.")
    parser.add_argument("--frame_ratio", type=int, default=1, help="Frame ratio to select images. Default is 100.")
    
    args = parser.parse_args()
    create_horizontal_strip(args.input_folder, args.output_file, args.n, args.frame_ratio) 