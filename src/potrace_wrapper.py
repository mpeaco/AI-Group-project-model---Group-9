import subprocess
import tempfile
import os


def bitmap_to_vector(image_path):
    """
    Wrapper to use the potrace lib

    Args:
        image_path (string): path of processed image
    """
    with tempfile.TemporaryDirectory() as tempdir:
        pbm_path = os.path.join(tempdir, "processed_bitmap_image.pbm")
        subprocess.run(
            ["magick", image_path, "-threshold", "50%", pbm_path], check=True
        )
        output_path = os.path.splitext(image_path)[0] + ".svg"
        subprocess.run(["potrace", pbm_path, "-s", "-o", output_path], check=True)

    return output_path
