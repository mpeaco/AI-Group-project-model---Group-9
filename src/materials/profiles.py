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

def add_custom_material(material_id, profile):
    """Add new material"""
    MATERIAL_PROFILES[material_id] = profile

def get_materials_by_type(material_type):
    """Get materials of same type"""
    results = {}
    for key, profile in MATERIAL_PROFILES.items():
        if profile['material_type'] == material_type:
            results[key] = profile
    return results

def calculate_cut_time(path_length_mm, material_profile):
    """ How long cutting will take"""
    speed = material_profile['cut_speed']
    passes = material_profile['cut_passes']
    pierce_time = material_profile['pierce_time']
    
    # Time = (distance / speed) * passes + pierce time
    cut_time_minutes = (path_length_mm / speed) * passes
    total_time = cut_time_minutes + (pierce_time / 60)  # Convert pierce time to minutes
    
    return total_time

def suggest_material_profile(detected_material, detected_thickness):
    """
    Suggest the best material profile based on ML detection results
    
    Args:
        detected_material: Material type from ML model ('cardboard', 'fabric', etc.)
        detected_thickness: Estimated thickness in mm
        
    Returns:
        Dictionary with suggested profile and alternatives
    """
    suggestions = {
        'primary': None,
        'alternatives': [],
        'confidence': 'medium'
    }
    
    # Material type mapping from ML classes to profile types
    ml_to_profile_map = {
        'cardboard': 'paper',
        'fabric': 'fabric', 
        'leather': 'leather',
        'metal': 'metal',
        'paper': 'paper',
        'wood': 'wood'
    }
    
    profile_type = ml_to_profile_map.get(detected_material, 'unknown')
    
    if profile_type == 'unknown':
        return suggestions
    
    # Get all profiles of the detected type
    matching_profiles = get_materials_by_type(profile_type)
    
    if not matching_profiles:
        return suggestions
    
    # Find the best match based on thickness
    best_match = None
    best_score = float('inf')
    alternatives = []
    
    for profile_id, profile in matching_profiles.items():
        profile_thickness = profile['thickness']
        thickness_diff = abs(profile_thickness - detected_thickness)
        
        # Score based on thickness difference
        score = thickness_diff
        
        if score < best_score:
            if best_match:
                alternatives.append({
                    'id': best_match[0],
                    'profile': best_match[1],
                    'thickness_diff': best_score
                })
            best_match = (profile_id, profile)
            best_score = score
        else:
            alternatives.append({
                'id': profile_id,
                'profile': profile,
                'thickness_diff': score
            })
    
    if best_match:
        suggestions['primary'] = {
            'id': best_match[0],
            'profile': best_match[1],
            'thickness_diff': best_score
        }
        
        # Sort alternatives by thickness difference
        alternatives.sort(key=lambda x: x['thickness_diff'])
        suggestions['alternatives'] = alternatives[:3]  # Top 3 alternatives
        
        # Determine confidence based on thickness match
        if best_score <= 0.5:  # Very close match
            suggestions['confidence'] = 'high'
        elif best_score <= 1.5:  # Reasonable match
            suggestions['confidence'] = 'medium'
        else:  # Poor match
            suggestions['confidence'] = 'low'
    
    return suggestions

def get_profile_summary(profile):
    """
    Get a human-readable summary of a material profile
    
    Args:
        profile: Material profile dictionary
        
    Returns:
        String summary of the profile
    """
    summary = f"{profile['name']} ({profile['thickness']}mm)"
    summary += f" - Cut: {profile['cut_speed']}mm/min @ {profile['cut_power']}%"
    
    if profile['cut_passes'] > 1:
        summary += f" x{profile['cut_passes']} passes"
    
    return summary

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
        'wood_thin': 'plywood_3mm',
        'wood_medium': 'plywood_6mm', 
        'wood_thick': 'plywood_6mm',
        'acrylic_thin': 'acrylic_3mm',
        'acrylic_medium': 'acrylic_5mm',
        'acrylic_thick': 'acrylic_5mm',
        'cardboard': 'cardboard_3mm',
        'fabric': 'felt_3mm',
        'leather': 'leather_2mm',
        'unknown': None
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

def smart_material_detection(image_path=None):
    """
    Smart material detection that tries ML first, then falls back to manual
    
    Args:
        image_path: Path to image file (optional)
        
    Returns:
        dict: Material profile
    """
    # Try ML detection first if image is provided
    if image_path:
        ml_result = detect_material_with_ml(image_path)
        if ml_result:
            profile = ml_material_to_profile(ml_result)
            if profile:
                print(f"✅ ML detected: {ml_result['material_name']} ({ml_result['thickness']:.1f}mm) "
                      f"- Confidence: {ml_result['confidence']:.1%}")
                return profile
    
    # Fall back to manual selection
    print("Using manual material selection...")
    from ..materials.workflow import choose_material
    material_id = choose_material()
    return get_material_profile(material_id)
