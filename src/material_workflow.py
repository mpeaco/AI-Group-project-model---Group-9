import json
from material_profiles import get_material_list, get_material_profile
from cutting_depth_manager import CuttingDepthManager, DEPTH_PRESETS

# This function lets user choose a material from the list
def choose_material():
    """Pick a material"""
    print("\nChoose Material:")
    materials = get_material_list()
    
    # show all materials with their thickness
    for i, mat in enumerate(materials):
        print(str(i+1) + ". " + mat['name'] + " (" + str(mat['thickness']) + "mm)")
    
    while True:
        try:
            choice = int(input("\nNumber: ")) - 1
            if 0 <= choice < len(materials):
                return materials[choice]['id']
            print("Try again.")
        except ValueError:
            print("Enter a number.")

def setup_cutting(paths, material_id):
    """Setup cutting"""
    print("\nCutting Setup:")
    
    # get the material profile and create manager
    profile = get_material_profile(material_id)
    manager = CuttingDepthManager(profile)
    
    print("Material: " + profile['name'] + " - " + str(len(paths)) + " paths")
    
    # get all the cutting presets
    presets = DEPTH_PRESETS
    operations = list(presets.items())
    
    print("\nOperations:")
    # display all the operations that can be done
    for i, (key, preset) in enumerate(operations):
        print(str(i+1) + ". " + preset['name'] + " (" + str(preset['depth_percentage']) + "%)")
    
    while True:
        try:
            choice = int(input("\nNumber: ")) - 1
            if 0 <= choice < len(operations):
                preset = operations[choice][1]
                # add the chosen operation to the manager
                manager.add_cut_operation(paths, preset['operation_type'], preset['depth_percentage'])
                return manager
            print("Try again.")
        except ValueError:
            print("Enter a number.")

def show_summary(manager):
    """Show summary"""
    print("\nSummary:")
    info = manager.get_material_usage_info()
    
    # show basic info about material and time
    print("Material: " + info['material_name'])
    print("Time: " + str(round(info['estimated_time'], 1)) + " min")
    
    # show cutting steps 
    sequence = manager.get_cutting_sequence()
    print("Steps: " + str(len(sequence)))
    for i, step in enumerate(sequence[:3]):
        print("  " + str(i+1) + ". " + step['operation'] + " - " + str(step['power']) + "%")
    if len(sequence) > 3:
        print("  ... and " + str(len(sequence)-3) + " more")

def run_workflow(paths, output_folder):
    """Main workflow"""
    print("\nLaser Cutting Setup")
    
    # Step 1: Choose material
    material_id = choose_material()
    # Step 2: Setup cutting operations
    manager = setup_cutting(paths, material_id)
    # Step 3: Show the summary
    show_summary(manager)
    
    # Ask if user wants to proceed
    if input("\nProceed? (y/n): ").lower() == 'y':
        save_settings(manager, material_id, output_folder)
        return manager
    # If user doesn't want to proceed, return None
    return None

def save_settings(manager, material_id, output_folder):
    """Save settings"""
    settings_file = output_folder + "/cutting_settings.json"
    
    # This converts complex objects to simple dictionaries
    def simplify_obj(obj):
        if hasattr(obj, '__dict__'):
            return {k: simplify_obj(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [simplify_obj(item) for item in obj]
        elif hasattr(obj, 'length'):
            return {'path_length': obj.length(), 'points': len(getattr(obj, 'points', []))}
        return obj
    
    # Get the cutting sequence and simplify it
    sequence = manager.get_cutting_sequence()
    simple_sequence = []
    for step in sequence:
        simple_step = step.copy()
        if 'paths' in simple_step:
            simple_step['paths'] = [simplify_obj(p) for p in simple_step['paths']]
        simple_sequence.append(simple_step)
    
    # Put everything in a dictionary for json
    settings = {
        'material_id': material_id,
        'material_profile': get_material_profile(material_id),
        'operations': simple_sequence,
        'summary': manager.get_material_usage_info()
    }
    
    # Write to file - indent=2 makes it readable
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print("Saved to: " + settings_file)
