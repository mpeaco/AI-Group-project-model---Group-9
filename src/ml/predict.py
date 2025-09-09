"""
Material prediction for laser cutting project
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import load_model

class MaterialRecognition:
    def __init__(self):
        self.materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self._load_model()
    
    def _load_model(self):
        model_path = "models_trained/material_classifier.pth"
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.model.eval()
                print(f"Model loaded: {model_path}")
            except:
                self.model = None
                print("Model loading failed")
        else:
            self.model = None
            print("No model found")
    def predict_material(self, image_path):
        if self.model:
            return self._neural_predict(image_path)
        else:
            return self._basic_predict(image_path)
    
    def _neural_predict(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                
                material = self.materials[pred.item()]
                conf_score = confidence.item()
                thickness = self._get_thickness(material)
                
                return {
                    'material_type': material,
                    'thickness_mm': thickness,
                    'confidence': conf_score,
                    'success': True
                }
        except:
            return self._basic_predict(image_path)
    
    def _basic_predict(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'material_type': 'unknown', 'thickness_mm': 3.0, 'confidence': 0.0, 'success': False}
            
            brightness = np.mean(img)
            
            if brightness > 200:
                return {'material_type': 'paper', 'thickness_mm': 0.2, 'confidence': 0.6, 'success': True}
            elif brightness > 150:
                return {'material_type': 'cardboard', 'thickness_mm': 2.0, 'confidence': 0.6, 'success': True}
            elif brightness > 100:
                return {'material_type': 'wood', 'thickness_mm': 6.0, 'confidence': 0.6, 'success': True}
            elif brightness > 80:
                return {'material_type': 'fabric', 'thickness_mm': 0.8, 'confidence': 0.6, 'success': True}
            elif brightness > 60:
                return {'material_type': 'leather', 'thickness_mm': 2.5, 'confidence': 0.6, 'success': True}
            else:
                return {'material_type': 'metal', 'thickness_mm': 1.5, 'confidence': 0.6, 'success': True}
        except:
            return {'material_type': 'unknown', 'thickness_mm': 3.0, 'confidence': 0.0, 'success': False}
    
    def _get_thickness(self, material):
        thickness = {
            'cardboard': 2.0, 'fabric': 0.8, 'leather': 2.5,
            'metal': 1.5, 'paper': 0.2, 'wood': 6.0
        }
        return thickness.get(material, 3.0)

def recognize_material_from_image(image_path):
    recognizer = MaterialRecognition()
    return recognizer.predict_material(image_path)

