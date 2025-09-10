# material prediction stuff
# kinda messy but works

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import load_model

"""The study inspired by BLOrange-AMD
https://github.com/pytorch/examples/blob/main/imagenet/main.py#L321"""
class MaterialRecognition:
    def __init__(self):
        # the 6 materials we can detect
        self.materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # image preprocessing - just normalize to [-1,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.load_trained_model()
    
    def load_trained_model(self):
        # try to load the trained model
        model_path = "models_trained/material_classifier.pth"
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.model.eval()
                print(f"loaded model from {model_path}")
            except Exception as e:
                print(f"failed to load model: {e}")
                self.model = None
        else:
            print("no trained model found, using fallback")
            self.model = None
    
    def predict_material(self, img_path):
        # main prediction function
        if self.model is not None:
            return self.predict_with_model(img_path)
        else:
            return self.predict_simple(img_path)
    
    def predict_with_model(self, img_path):
        # use the neural network
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probs, 1)
                
                material_name = self.materials[prediction.item()]
                conf_val = confidence.item()
                thickness = self.get_material_thickness(material_name)
                
                return {
                    'material_type': material_name,
                    'thickness_mm': thickness,
                    'confidence': conf_val,
                    'success': True
                }
        except Exception as e:
            print(f"model prediction failed: {e}")
            return self.predict_simple(img_path)
    
    def predict_simple(self, img_path):
        # fallback method - just look at brightness
        try:
            img = cv2.imread(img_path)
            if img is None:
                return self.default_result()
            
            # calculate average brightness
            avg_brightness = np.mean(img)
            
            # simple thresholding based on brightness
            # lighter = paper/cardboard, darker = metal/leather
            if avg_brightness > 200:
                mat = 'paper'
                thick = 0.2
            elif avg_brightness > 150:
                mat = 'cardboard' 
                thick = 2.0
            elif avg_brightness > 100:
                mat = 'wood'
                thick = 6.0
            elif avg_brightness > 80:
                mat = 'fabric'
                thick = 0.8
            elif avg_brightness > 60:
                mat = 'leather'
                thick = 2.5
            else:
                mat = 'metal'
                thick = 1.5
                
            return {
                'material_type': mat,
                'thickness_mm': thick,
                'confidence': 0.6,  # low confidence for simple method
                'success': True
            }
        except:
            return self.default_result()
    
    def get_material_thickness(self, material):
        # typical thickness values for each material
        thicknesses = {
            'cardboard': 2.0, 
            'fabric': 0.8, 
            'leather': 2.5,
            'metal': 1.5, 
            'paper': 0.2, 
            'wood': 6.0
        }
        return thicknesses.get(material, 3.0)  # default 3mm
    
    def default_result(self):
        # return this when everything fails
        return {
            'material_type': 'unknown', 
            'thickness_mm': 3.0, 
            'confidence': 0.0, 
            'success': False
        }

# simple function for external use
def recognize_material_from_image(img_path):
    recognizer = MaterialRecognition()
    return recognizer.predict_material(img_path)

