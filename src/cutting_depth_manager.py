from materials.profiles import get_material_profile
from processing.path_optimisation import CuttingPath

class CuttingDepthManager:
    """Manages cutting depth, multi-pass cutting, and operation types"""
    
    def __init__(self, material_profile):
        self.material = material_profile
        self.operations = []
        
    def add_cut_operation(self, paths, operation_type="cut", depth_percentage=100):
        """
        Add cutting operation with specified depth
        
        Args:
            paths: List of CuttingPath objects
            operation_type: "cut", "engrave", "score", "mark"
            depth_percentage: 0-100% of material thickness
        """
        # Calculate passes needed
        passes_needed = 1  # Default value
        if operation_type == "cut":
            if depth_percentage > 75:
                passes_needed = self.material.get('cut_passes', 1)
            elif depth_percentage > 50:
                passes_needed = max(1, self.material.get('cut_passes', 1) // 2)
        
        # Calculate power
        power = self.material['cut_power']  # Default power
        if operation_type == "engrave":
            power = self.material['engrave_power']
        elif operation_type == "score" or operation_type == "mark":
            power = self.material['engrave_power'] * 0.7
        elif passes_needed > 1:
            power = min(100, self.material['cut_power'] * 0.8)
        
        # Calculate speed
        speed = self.material['cut_speed']  # Default speed
        if operation_type == "engrave":
            speed = self.material['engrave_speed']
        elif operation_type == "score" or operation_type == "mark":
            speed = self.material['engrave_speed'] * 0.8
        
        # Create operation
        operation = {
            'type': operation_type,
            'paths': paths,
            'depth_percentage': depth_percentage,
            'passes_needed': passes_needed,
            'power_per_pass': power,
            'speed': speed
        }
        
        self.operations.append(operation)
        return operation
    
    def get_cutting_sequence(self):
        """Get optimized sequence of all operations"""
        # Sort operations by depth (shallow first, then deep)
        sorted_ops = []
        for op in self.operations:
            sorted_ops.append(op)
        
        sorted_ops.sort(key=lambda x: x['depth_percentage'])
        
        cutting_sequence = []
        
        for op in sorted_ops:
            for pass_num in range(op['passes_needed']):
                sequence_step = {
                    'operation': op['type'],
                    'pass_number': pass_num + 1,
                    'total_passes': op['passes_needed'],
                    'paths': op['paths'],
                    'power': op['power_per_pass'],
                    'speed': op['speed'],
                    'depth_percentage': op['depth_percentage']
                }
                cutting_sequence.append(sequence_step)
        
        return cutting_sequence
    
    def estimate_total_time(self):
        """Estimate total cutting time for all operations"""
        total_time = 0
        
        for op in self.operations:
            # Calculate total path length
            total_length = 0
            for path in op['paths']:
                total_length += path.length()
            
            # Time per pass
            speed = op['speed']  # mm/min
            time_per_pass = total_length / speed
            
            # Pierce time per path per pass
            pierce_time = len(op['paths']) * self.material['pierce_time'] / 60
            
            # Total time for this operation
            op_time = (time_per_pass + pierce_time) * op['passes_needed']
            total_time += op_time
        
        return total_time
    
    def get_material_usage_info(self):
        """Get information about material usage and waste"""
        info = {
            'material_name': self.material['name'],
            'thickness': self.material['thickness'],
            'total_operations': len(self.operations),
            'kerf_compensation': self.material['kerf_width'],
            'estimated_time': self.estimate_total_time()
        }
        
        # Count operation types
        op_types = {}
        for op in self.operations:
            op_type = op['type']
            if op_type not in op_types:
                op_types[op_type] = {'count': 0, 'paths': 0}
            
            op_types[op_type]['count'] += 1
            op_types[op_type]['paths'] += len(op['paths'])
        
        info['operation_breakdown'] = op_types
        return info

# Cutting depth presets
DEPTH_PRESETS = {
    "full_cut": {
        "name": "Full Cut Through",
        "depth_percentage": 100,
        "operation_type": "cut"
    },
    "half_cut": {
        "name": "Half Depth Cut", 
        "depth_percentage": 50,
        "operation_type": "cut"
    },
    "score_line": {
        "name": "Score Line",
        "depth_percentage": 25,
        "operation_type": "score"
    },
    "engrave_deep": {
        "name": "Deep Engrave",
        "depth_percentage": 15,
        "operation_type": "engrave"
    },
    "engrave_light": {
        "name": "Light Engrave",
        "depth_percentage": 5,
        "operation_type": "engrave"
    },
    "mark_only": {
        "name": "Surface Mark",
        "depth_percentage": 2,
        "operation_type": "mark"
    }
}
