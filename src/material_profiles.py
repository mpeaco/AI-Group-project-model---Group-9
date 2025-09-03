# Material profiles for laser cutting
# Each material has specific cutting parameters

MATERIAL_PROFILES = {
    # Paper & Cardboard
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
        "material_type": "paper"
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
    
    "plywood_6mm": {
        "name": "Plywood 6mm", 
        "thickness": 6.0,
        "cut_speed": 400,
        "cut_power": 95,
        "cut_passes": 2,  # Multiple passes needed
        "engrave_speed": 800,
        "engrave_power": 65,
        "kerf_width": 0.2,
        "pierce_time": 1.0,
        "material_type": "wood"
    },
    
    "mdf_3mm": {
        "name": "MDF 3mm",
        "thickness": 3.0,
        "cut_speed": 500,
        "cut_power": 85,
        "cut_passes": 1,
        "engrave_speed": 900,
        "engrave_power": 60,
        "kerf_width": 0.18,
        "pierce_time": 0.6,
        "material_type": "wood"
    },
    
    # Acrylic Materials
    "acrylic_3mm": {
        "name": "Acrylic 3mm",
        "thickness": 3.0,
        "cut_speed": 300,
        "cut_power": 70,  # Good for clear/white acrylic
        "cut_passes": 1,
        "engrave_speed": 1500,
        "engrave_power": 40,
        "kerf_width": 0.1,
        "pierce_time": 0.3,
        "material_type": "acrylic",
        "notes": "Reduce power to 55-65% for colored acrylic"
    },
    
    "acrylic_5mm": {
        "name": "Acrylic 5mm",
        "thickness": 5.0,
        "cut_speed": 200,  # Slow speed for smooth edge
        "cut_power": 90,   # High power to prevent striations
        "cut_passes": 1,
        "engrave_speed": 1200,
        "engrave_power": 50,
        "kerf_width": 0.12,
        "pierce_time": 0.8,
        "material_type": "acrylic",
        "notes": "Very slow speed prevents visible striations on edge"
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
