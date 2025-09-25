# AI-Group-project-model---Group-9

This Repository contains the source code and assessment documents for the MSc Artifcial Intelligence group 9 project.

## Image to Laser Cutting

An end-to-end system to take photos or hand drawn sketches and make them useable by a laser cutter without the need for various different software at each step of the process.

## Installation

During testing the repo can be cloned and the program can be run from terminal

## Usage

`pip install -r requirements.txt`

To run the main program the main.py file is located in the `src` directory. In the terminal `python3 main.py'.

The image processing can be done using opencv or pyvips. Pyvips can be used by passing the `-p` flag when running the main.py file.

### Alternative Run Methods

For convenience, you can also run the project using the provided scripts:

- On Windows using Command Prompt: Double-click `run_project.bat`
- On Windows using PowerShell: Right-click `run_project.ps1` and select "Run with PowerShell"

These scripts automatically navigate to the source directory and run the program using the project's virtual environment.

## Dependencies

./venv/Scripts/Activate.ps1


## To train the model
cd C:\Users\sangn\OneDrive\Documents\GitHub\AI-Group-project-model---Group-9\src\ml

then run:

C:/Users/sangn/OneDrive/Documents/GitHub/AI-Group-project-model---Group-9/venv/Scripts/python.exe train.py