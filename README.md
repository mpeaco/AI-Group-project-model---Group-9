# AI-Group-project-model---Group-9

This Repository contains the source code and assessment documents for the MSc Artifcial Intelligence group 9 project.

## Image to Laser Cutting

An end-to-end system to take photos or hand drawn sketches and make them useable by a laser cutter without the need for various different software at each step of the process.

## Installation

During testing the repo can be cloned and the program can be run from terminal

### Optional External Dependencies

The project works fully with the default OpenCV image processing. However, if you want to use additional features, you may need to install:

**For PyVIPS Image Processing:**
- If you choose PyVIPS during runtime, you'll need to install VIPS binaries separately
- Download VIPS from: https://github.com/libvips/libvips/releases
- PyVIPS may show warnings about missing modules, but core functionality will work

**For Advanced Vector Processing (Optional):**
- **Potrace** - For bitmap to vector conversion (alternative method)
- **ImageMagick** - Used with Potrace for image preprocessing

Note: The project runs successfully with just the Python packages from `requirements.txt`. External dependencies are optional and provide alternative processing methods.

## Usage

### Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the program:
   ```bash
   cd src
   python main.py
   ```

### Running the Program

The main program is located in the `src` directory. When you run `python main.py`, the system will:

1. **Ask about image processing method**: Choose between OpenCV (default) or PyVIPS
   - Press `n` for OpenCV (recommended)
   - Press `y` for PyVIPS (if you have it installed)

2. **Process your image**: The system automatically loads a sample image and converts it to laser-cutting paths

3. **Select material**: Choose from available materials:
   - Cardboard 3mm
   - Paper 160gsm  
   - Plywood 3mm
   - Felt 2mm
   - Leather 2mm
   - Thin Metal Sheet

4. **Choose cutting operation**: Select the type of cut you want:
   - Full Cut Through (100%)
   - Half Depth Cut (50%)
   - Score Line (25%)
   - Deep Engrave (15%)
   - Light Engrave (5%)
   - Surface Mark (2%)

5. **Get your files**: The system generates multiple output formats in a timestamped folder located at `src/sample_data/sample_results/output_results_[timestamp]/`:
   - `.svg` - Vector graphics file
   - `.dxf` - CAD file for laser cutters
   - `.pbm` - Processed bitmap
   - Visualization images
   - Cutting settings JSON

### Alternative Run Methods

For convenience, you can also run the project using the provided scripts:

- **Windows Command Prompt**: Double-click `run_project.bat`
- **Windows PowerShell**: Double-click `run_project.ps1`

These scripts automatically navigate to the source directory and run the program using the project's virtual environment.

## Dependencies

./venv/Scripts/Activate.ps1


## To train the model
cd C:\Users\sangn\OneDrive\Documents\GitHub\AI-Group-project-model---Group-9\src\ml

then run:

C:/Users/sangn/OneDrive/Documents/GitHub/AI-Group-project-model---Group-9/venv/Scripts/python.exe train.py