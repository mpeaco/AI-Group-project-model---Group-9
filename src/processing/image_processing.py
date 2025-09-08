from pathlib import Path
import pyvips
import numpy as np
import cv2 as cv
from svgpathtools import svg2paths


def image_processing_pyvips(image):
    image = image.gaussblur(1.5)
    if image.hasalpha():
        image = image.flatten(background=[255]).unpremultiply()
    image = image.colourspace("b-w")
    if image.bands > 1:
        image = image.extract_band(0)
    
    image = image.copy(interpretation="b-w")
    image = convert_pyvips_to_numpy(image)
    _, otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(otsu, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
    
    return cleaned


def image_processing_opencv(image):
    """
    Takes an image and converts to greyscale
    perform smoothing
    threshold
    """

    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 1.4)
    
    # Try Otsu's automatic thresholding for better results
    ret, img_thresholded = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(f"Otsu threshold value: {ret}")
    
    # Use smaller kernel to preserve more detail
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(img_thresholded, cv.MORPH_OPEN, kernel)

    # Fill small holes
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
    return cleaned


def edge_detection(img):
    canny_edges = cv.Canny(img, 100, 300)

    return canny_edges


def line_extraction(img):
    """
    Extracts contours using opencv findContours

    Args:
        img (ndarray): processed image

    Returns:
        list: contours found
    """
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("Contours found: {}", len(contours))
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 20:
            filtered_contours.append(contour)
            print(f"Kept contour with area: {area}")
    print("After filtering: {}".format(len(filtered_contours)))
    return filtered_contours


def create_bitmap(img, output_path=None):
    """
    Needs to create a bitmap from the processed image to save to specified path

    Args:
        img (numpy.ndarray): the processed image to be stored
        output_path (str, optional): Path to save the bitmap file
        
    Returns:
        str: Path to the saved bitmap file
    """
    if output_path is None:
        output_filename = "processed_bitmap_image.pbm"
    else:
        output_filename = output_path
        
    # Make sure the directory exists
    import os
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    cv.imwrite(output_filename, img)
    return output_filename


def get_paths(image_path):
    paths, _ = svg2paths(image_path)
    if paths:
        print("found paths")
        print("Number of paths found {}".format(len(paths)))
        print("Type of Paths: {}".format(type(paths[0])))
    return paths

def convert_pyvips_to_numpy(image):
    """
    Converts a pyvips.Image to a numpy array

    Args:
        image (pyvips.Image): image

    Returns:
        numpy.ndarray: numpy array of image
    """
    pyvips_im = image
    height = image.height
    width = image.width
    bands = image.bands
    print("It is a pyvips image")
    image = np.frombuffer(pyvips_im.write_to_memory(), dtype=np.uint8)
    np_image = image.reshape(height, width, bands)

    return np_image

# ML Image Processing Functions

def preprocess_for_material_detection(image_path, output_path=None):
    """Prepares an image for material classification model input.
    
    Takes a source image, does necessary preprocessing (resize, denoise, 
    enhance contrast) to make it suitable for ML model input.
    
    Args:
        image_path: Source image location
        output_path: Where to save the processed image (optional)
        
    Returns:
        Path to the processed image as string
    """
    try:
        # Read the image
        img = cv.imread(str(image_path))
        if img is None:
            print(f"Warning: Couldn't open {image_path}")
            return str(image_path)
        
        # Color conversion for processing
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Standard ML input size
        img_224 = cv.resize(rgb, (224, 224))
        
        # Denoise but preserve important features
        denoised = cv.fastNlMeansDenoisingColored(img_224, None, 10, 10, 7, 21)
        
        # Boost contrast to help material features stand out
        lab = cv.cvtColor(denoised, cv.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv.split(lab)
        
        # Apply contrast enhancement only to luminance channel
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_chan)
        
        # Merge channels back
        enhanced = cv.merge([l_enhanced, a_chan, b_chan])
        rgb_enhanced = cv.cvtColor(enhanced, cv.COLOR_LAB2RGB)
        
        # Generate output path if needed
        if not output_path:
            name = Path(image_path).stem
            output_path = Path(image_path).parent / f"{name}_ml_processed.jpg"
        
        # Save as BGR (OpenCV format)
        cv.imwrite(str(output_path), cv.cvtColor(rgb_enhanced, cv.COLOR_RGB2BGR))
        
        print(f"Image processed: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return str(image_path)

def extract_material_region(image_path, roi_percentage=0.8):
    """Crops image to focus on the central region with material.
    
    Most material samples are in the center of photos, so this
    removes edges which often contain background noise.
    
    Args:
        image_path: Path to image
        roi_percentage: How much of center to keep (0.0-1.0)
        
    Returns:
        Path to cropped image
    """
    try:
        img = cv.imread(str(image_path))
        if img is None:
            return str(image_path)
        
        # Get dimensions
        height, width = img.shape[:2]
        
        # Calculate center crop size
        crop_height = int(height * roi_percentage)
        crop_width = int(width * roi_percentage)
        
        # Find top-left corner for centered crop
        y_start = (height - crop_height) // 2
        x_start = (width - crop_width) // 2
        
        # Extract the region
        cropped = img[y_start:y_start+crop_height, x_start:x_start+crop_width]
        
        # Save result
        name = Path(image_path).stem
        out_path = Path(image_path).parent / f"{name}_material_region.jpg"
        cv.imwrite(str(out_path), cropped)
        
        return str(out_path)
        
    except Exception as e:
        print(f"Crop failed: {str(e)}")
        return str(image_path)

def prepare_image_for_ml_pipeline(image_path, output_folder=None):
    """Runs full image preparation workflow for material detection.
    
    Creates necessary directories, manages file paths, and 
    coordinates the complete preprocessing pipeline.
    
    Args:
        image_path: Source image
        output_folder: Where to store processed versions
        
    Returns:
        Dictionary with paths to all processed versions
    """
    # Set up output location
    if output_folder is None:
        output_folder = Path(image_path).parent / "ml_preprocessed"
    else:
        output_folder = Path(output_folder)
    
    # Create folder if needed
    output_folder.mkdir(exist_ok=True)
    
    # Track results
    results = {
        'original': str(image_path),
        'material_region': None,
        'ml_ready': None,
        'success': False
    }
    
    try:
        print(f"Processing: {Path(image_path).name}")
        
        # First crop to region of interest
        cropped = extract_material_region(image_path)
        results['material_region'] = cropped
        
        # Then apply ML-specific preprocessing
        name = Path(image_path).stem
        ml_ready_path = output_folder / f"{name}_ml_ready.jpg"
        processed = preprocess_for_material_detection(cropped, ml_ready_path)
        results['ml_ready'] = processed
        
        results['success'] = True
        print(f"✅ Processing pipeline complete")
        return results
        
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        results['ml_ready'] = str(image_path)  # Fallback to original
        return results

def enhanced_image_processing_with_ml(image_path, use_ml=True, output_folder=None):
    """Combines image processing with material detection.
    
    Main function that orchestrates the complete workflow from
    image preparation through material detection.
    
    Args:
        image_path: Source image
        use_ml: Whether to use ML detection
        output_folder: Output directory
        
    Returns:
        Results dict with material info and processed images
    """
    results = {
        'processed_image': None,
        'material_info': None,
        'ml_used': False,
        'preprocessing_paths': None
    }
    
    try:
        # Run ML pipeline if requested
        if use_ml:
            paths = prepare_image_for_ml_pipeline(image_path, output_folder)
            results['preprocessing_paths'] = paths
            
            if paths['success']:
                # Import here to avoid circular dependencies
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                from materials.profiles import detect_material_with_ml
                material_data = detect_material_with_ml(paths['ml_ready'])
                
                if material_data:
                    results['material_info'] = material_data
                    results['ml_used'] = True
                    print(f"✅ Material detected: {material_data['material_name']} "
                          f"({material_data['thickness']:.1f}mm)")
        
        # Additional processing could go here
        # (standard image processing for cutting paths)
        
        return results
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return results
