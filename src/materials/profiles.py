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
