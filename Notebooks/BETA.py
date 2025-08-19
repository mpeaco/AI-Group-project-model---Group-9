"""
TODO
-add our contours generator
-test different ai pathfinders for stitching the lines together
-find a way to smooth curves ?
"""

import cv2
import numpy as np

def writeFile(filename, imagesize, points):
    width, height = imagesize
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" version="1.1">\n'
    svg_footer = '</svg>'
    svg_body = ""

    for contour in points:
        path_data = "M " + " L ".join(f"{pt[0]} {pt[1]}" for pt in contour) + " Z"
        svg_body += f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="1"/>\n'

    with open(filename, "w") as f:
        f.write(svg_header + svg_body + svg_footer)
    print(f"SVG saved to {filename}")


def contoursToList(contours, svg_filename):
    output = []
    for contour in contours:
        temp = contour[0]
        for pt in contour:
            output.append([temp[0], pt[0]])
            temp = pt
        
        output.append([temp[0], contour[0][0]])
    
    output = np.array(output)
     
    return output


# TEMP:
# waiting on mark to send the best version from his findings,
# this is from the internet
def image_to_contours(image_path, boundary):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    _, thresh = cv2.threshold(img, boundary, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape
    return contours, (width, height)

if __name__ == "__main__":
    input_image = "AI-Group-project-model---Group-9/Notebooks/sample_data/sunflower.jpg"  # replace with your image path
    output_svg = "AI-Group-project-model---Group-9/Notebooks/sample_data/output2.svg"

#    for x in range(255):
 #       output_svg = "AI-Group-project-model---Group-9/Notebooks/sample_data/output" + str(x) + ".svg"
 
    contours, size = image_to_contours(input_image, 127)
    guy = contoursToList(contours, output_svg)
    writeFile(output_svg, size, guy)
