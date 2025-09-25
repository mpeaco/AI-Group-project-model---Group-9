import ezdxf
from svgpathtools import svg2paths
from pathlib import Path

def svg_to_dxf(svg_path: str, output_path: str = None) -> str:
    """
    Convert SVG file to DXF format for laser cutting
    
    Args:
        svg_path (str): Path to input SVG file
        output_path (str, optional): Path for output DXF file. Defaults to None.
    
    Returns:
        str: Path to created DXF file
    """
    # Create new DXF document
    doc = ezdxf.new('R2010')  # AutoCAD 2010 format
    msp = doc.modelspace()
    
    # Get paths from SVG
    paths, _ = svg2paths(svg_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = Path(svg_path).with_suffix('.dxf')
    
    # Convert each SVG path to DXF polyline
    for path in paths:
        points = []
        # Sample points along the path
        for segment in path:
            points.append((segment.start.real, segment.start.imag))
            points.append((segment.end.real, segment.end.imag))
            
        # Create polyline in DXF
        if points:
            msp.add_lwpolyline(points)
    
    # Save DXF file
    doc.saveas(output_path)
    return str(output_path)

def convert_all_svg_in_directory(input_dir: str, output_dir: str = None):
    """
    Convert all SVG files in a directory to DXF format
    
    Args:
        input_dir (str): Directory containing SVG files
        output_dir (str, optional): Directory for output DXF files. Defaults to input directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all SVG files
    for svg_file in input_path.glob('*.svg'):
        output_file = output_path / f"{svg_file.stem}.dxf"
        try:
            svg_to_dxf(str(svg_file), str(output_file))
            print(f"Converted {svg_file.name} to {output_file.name}")
        except Exception as e:
            print(f"Error converting {svg_file.name}: {str(e)}")