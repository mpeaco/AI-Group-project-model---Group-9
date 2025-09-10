# Material profiles for laser cutting
# Each material has specific cutting parameters

MATERIAL_PROFILES = {
    # Cardboard & Paper
    "cardboard_3mm": {
        "name": "Cardboard 3mm",
        "thickness": 3.0,  # mm
        "cut_speed": 600,  # mm/min - reduced for reliable cutting
        "cut_power": 65,   # % laser power
        "cut_passes": 1,   # number of passes
        "engrave_speed": 1200,
        "engrave_power": 45,
        "kerf_width": 0.1,  
        "pierce_time": 0.2,  # seconds
        "material_type": "cardboard"
    },
    
    "paper_160gsm": {
        "name": "Paper 160gsm",
        "thickness": 0.2,
        "cut_speed": 900,  
        "cut_power": 35,
        "cut_passes": 1,
        "engrave_speed": 2000,
        "engrave_power": 25,  # Light and fast to avoid burning
        "kerf_width": 0.05,
        "pierce_time": 0.1,
        "material_type": "paper"
    },
    
    # Wood Materials  
    "plywood_3mm": {
        "name": "Plywood 3mm",
        "thickness": 3.0,
        "cut_speed": 600,
        "cut_power": 80,
        "cut_passes": 1,
        "engrave_speed": 1000,
        "engrave_power": 55,
        "kerf_width": 0.15,
        "pierce_time": 0.5,
        "material_type": "wood"
    },
    
    # Fabric & Leather
    "felt_2mm": {
        "name": "Felt 2mm",
        "thickness": 2.0,
        "cut_speed": 1200,  # Can go faster with high power machines
        "cut_power": 50,    # Reduced to avoid fire risk
        "cut_passes": 1,
        "engrave_speed": 1800,
        "engrave_power": 35,
        "kerf_width": 0.2,
        "pierce_time": 0.15,
        "material_type": "fabric",
        "notes": "High speed, low power with excellent air assist to prevent fire"
    },
    
    "leather_2mm": {
        "name": "Leather 2mm", 
        "thickness": 2.0,
        "cut_speed": 800,   # Higher speed for less charring
        "cut_power": 50,    # Lower power to reduce burning
        "cut_passes": 1,
        "engrave_speed": 1000,
        "engrave_power": 45,
        "kerf_width": 0.1,
        "pierce_time": 0.3,  # Reduced pierce time
        "material_type": "leather",
        "notes": "Higher speed, lower power reduces charred edge. Genuine vs synthetic behaves differently"
    },
    
    # Metal Materials (thin sheets suitable for laser cutting)
    "metal_thin": {
        "name": "Thin Metal Sheet",
        "thickness": 0.5,   # Very thin metal sheets only
        "cut_speed": 100,   # Very slow for precision
        "cut_power": 100,   # Maximum power needed
        "cut_passes": 3,    # Multiple passes required
        "engrave_speed": 500,
        "engrave_power": 80,
        "kerf_width": 0.05,
        "pierce_time": 2.0, # Longer pierce time
        "material_type": "metal",
        "notes": "Only very thin metal sheets. Requires special lens and high power. Use with extreme caution."
    }
}

def get_material_list():
    """Get materials for user to pick from"""
    materials = []
    for key, profile in MATERIAL_PROFILES.items():
        materials.append({
            'id': key,
            'name': profile['name'],
            'thickness': profile['thickness'],
            'type': profile['material_type']
        })
    return materials

def get_material_profile(material_id):
    """Get material info by ID"""
    if material_id in MATERIAL_PROFILES:
        return MATERIAL_PROFILES[material_id]
    return None

def get_materials_by_type(material_type):
    """Get materials of same type"""
    results = {}
    for key, profile in MATERIAL_PROFILES.items():
        if profile['material_type'] == material_type:
            results[key] = profile
    return results

# === ML Integration Functions ===

def detect_material_with_ml(image_path):
    """
    Use ML to detect material type and thickness from an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Contains material_name, thickness, confidence, or None if ML fails
    """
    try:
        # Import ML components
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ml.recognition import MaterialRecognition
        
        # Initialize ML recognizer
        recognizer = MaterialRecognition()
        
        # Check if ML model is available
        if not recognizer.is_available():
            print("ML model not available, falling back to manual selection")
            return None
        
        # Run ML detection
        result = recognizer.predict_material(image_path)
        
        if result and result.get('confidence', 0) > 0.7:  # Only accept high confidence
            return {
                'material_name': result['material_type'],
                'thickness': result['thickness_mm'],
                'confidence': result['confidence'],
                'method': 'ml_detection'
            }
        else:
            print(f"ML confidence too low ({result.get('confidence', 0):.2f}), falling back to manual")
            return None
            
    except Exception as e:
        print(f"ML detection failed: {e}")
        return None

def ml_material_to_profile(ml_result):
    """
    Convert ML detection result to a material profile
    
    Args:
        ml_result: Result from detect_material_with_ml()
        
    Returns:
        dict: Material profile or None if no match found
    """
    if not ml_result:
        return None
    
    material_name = ml_result['material_name']
    thickness = ml_result['thickness']
    
    # Map ML material types to our profile names
    material_mapping = {
        'cardboard': 'cardboard_3mm',
        'fabric': 'felt_2mm',
        'leather': 'leather_2mm',
        'metal': 'metal_thin',
        'paper': 'paper_160gsm',
        'wood': 'plywood_3mm'
    }
    
    profile_name = material_mapping.get(material_name)
    if not profile_name or profile_name not in MATERIAL_PROFILES:
        return None
    
    # Get the base profile
    profile = MATERIAL_PROFILES[profile_name].copy()
    
    # Adjust profile based on detected thickness
    profile = adjust_profile_for_thickness(profile, thickness)
    
    # Add ML detection info
    profile['ml_detected'] = True
    profile['ml_confidence'] = ml_result['confidence']
    profile['detected_thickness'] = thickness
    
    return profile

def adjust_profile_for_thickness(base_profile, detected_thickness):
    """
    Adjust cutting parameters based on detected thickness
    
    Args:
        base_profile: Base material profile
        detected_thickness: Thickness detected by ML (mm)
        
    Returns:
        dict: Adjusted profile
    """
    profile = base_profile.copy()
    base_thickness = profile['thickness']
    
    if detected_thickness <= 0:
        return profile
    
    # Calculate thickness ratio
    thickness_ratio = detected_thickness / base_thickness
    
    # Adjust parameters based on thickness
    if thickness_ratio > 1.5:  # Much thicker
        profile['cut_power'] = min(100, profile['cut_power'] * 1.2)
        profile['cut_speed'] = max(100, profile['cut_speed'] * 0.8)
        profile['cut_passes'] = profile.get('cut_passes', 1) + 1
        profile['pierce_time'] = profile['pierce_time'] * 1.5
    elif thickness_ratio > 1.2:  # Somewhat thicker
        profile['cut_power'] = min(100, profile['cut_power'] * 1.1)
        profile['cut_speed'] = max(100, profile['cut_speed'] * 0.9)
        profile['pierce_time'] = profile['pierce_time'] * 1.2
    elif thickness_ratio < 0.7:  # Much thinner
        profile['cut_power'] = max(30, profile['cut_power'] * 0.8)
        profile['cut_speed'] = min(2000, profile['cut_speed'] * 1.2)
        profile['pierce_time'] = profile['pierce_time'] * 0.7
    elif thickness_ratio < 0.9:  # Somewhat thinner  
        profile['cut_power'] = max(30, profile['cut_power'] * 0.9)
        profile['cut_speed'] = min(2000, profile['cut_speed'] * 1.1)
        profile['pierce_time'] = profile['pierce_time'] * 0.8
    
    # Update the thickness in profile
    profile['thickness'] = detected_thickness
    profile['name'] = f"{profile['name']} (ML: {detected_thickness}mm)"
    
    return profile
