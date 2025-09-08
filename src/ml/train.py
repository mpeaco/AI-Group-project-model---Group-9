"""
Simple Material Classification Training
Train a model to recognize different materials for laser cutting
// Working but not optimal // 
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import numpy as np
from models import MaterialNet, save_model, get_model_info
from sklearn.metrics import classification_report

class MaterialDataset(Dataset):
    """Simple dataset class for material images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

def load_data(data_dir):
    """Load images and create random train/test split"""
    # Material types we want to classify
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    
    # Create lists to store all image data
    all_images = []
    all_labels = []
    material_counts = {}
    
    print("Loading images...")
    
    # Load images from each material folder
    for i, material in enumerate(materials):
        folder_path = os.path.join(data_dir, material)
        if os.path.exists(folder_path):
            count = 0
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, filename)
                    all_images.append(img_path)
                    all_labels.append(i)  # Use index as label
                    count += 1
            material_counts[material] = count
    
    # Show what we loaded
    total = len(all_images)
    print(f"\nDataset loaded:")
    for material in materials:
        count = material_counts.get(material, 0)
        percent = (count / total) * 100 if total > 0 else 0
        print(f"  {material}: {count} images ({percent:.1f}%)")
    
    return all_images, all_labels, total
    
def create_train_test_split(all_images, all_labels):
    """Split data with equal samples per material (balanced sampling)"""
    
    # Define how many samples per material
    SAMPLES_PER_MATERIAL_TRAIN = 80
    SAMPLES_PER_MATERIAL_TEST = 20
    TOTAL_PER_MATERIAL = SAMPLES_PER_MATERIAL_TRAIN + SAMPLES_PER_MATERIAL_TEST
    
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    
    # Group images by material
    material_images = {i: [] for i in range(6)}
    for img_path, label in zip(all_images, all_labels):
        material_images[label].append(img_path)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    print(f"\nBalanced sampling ({SAMPLES_PER_MATERIAL_TRAIN} train + {SAMPLES_PER_MATERIAL_TEST} test per material):")
    
    # Sample equally from each material
    for i, material in enumerate(materials):
        available_images = material_images[i]
        
        if len(available_images) < TOTAL_PER_MATERIAL:
            print(f"  WARNING: {material} has only {len(available_images)} images, need {TOTAL_PER_MATERIAL}")
            # Use all available images
            np.random.shuffle(available_images)
            train_count = int(len(available_images) * 0.8)  # 80% train, 20% test
            material_train = available_images[:train_count]
            material_test = available_images[train_count:]
        else:
            # Randomly sample the required number
            np.random.shuffle(available_images)
            selected_images = available_images[:TOTAL_PER_MATERIAL]
            material_train = selected_images[:SAMPLES_PER_MATERIAL_TRAIN]
            material_test = selected_images[SAMPLES_PER_MATERIAL_TRAIN:]
        
        # Add to train set
        train_images.extend(material_train)
        train_labels.extend([i] * len(material_train))
        
        # Add to test set
        test_images.extend(material_test)
        test_labels.extend([i] * len(material_test))
        
        print(f"  {material}: Train={len(material_train)}, Test={len(material_test)}")
    
    print(f"\nBalanced split completed:")
    print(f"  Total Training: {len(train_images)} images")
    print(f"  Total Testing: {len(test_images)} images")
    
    return train_images, train_labels, test_images, test_labels

def create_data_loaders(train_images, train_labels, test_images, test_labels, batch_size=16):
    """Create data loaders for training and testing"""
    
    # Standard ImageNet normalization for better performance
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Standard ImageNet normalization for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MaterialDataset(train_images, train_labels, train_transform)
    test_dataset = MaterialDataset(test_images, test_labels, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def test_hyperparameters(train_images, train_labels, test_images, test_labels, epochs=3):
    """Test different hyperparameters systematically"""
    
    print("=" * 60)
    print("HYPERPARAMETER TESTING")
    print("=" * 60)
    
    # hyperparameter combinations 
    learning_rates = [0.001]  #  1 learning rate
    batch_sizes = [16, 32]    #  2 batch sizes
    
    results = []
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTesting LR: {lr}, Batch Size: {batch_size}")
            
            # Create data loaders with current batch size
            train_loader_test, test_loader_test = create_data_loaders(
                train_images, train_labels, test_images, test_labels, batch_size=batch_size
            )
            
            # Create new model
            model = MaterialNet(num_classes=6)
            
            # Train with current parameters - use only 3 epochs for quick testing
            try:
                trained_model, accuracy = train_model(model, train_loader_test, test_loader_test, 
                                                    epochs=epochs, learning_rate=lr)
                results.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'accuracy': accuracy
                })
                print(f"Result: {accuracy:.2f}% accuracy")
            except Exception as e:
                print(f"Failed with error: {e}")
                results.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'accuracy': 0.0
                })
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n" + "=" * 60)
    print("HYPERPARAMETER TEST RESULTS")
    print("=" * 60)
    for result in results:
        print(f"LR: {result['lr']}, Batch: {result['batch_size']}, Accuracy: {result['accuracy']:.2f}%")
    
    print(f"\nBest Parameters:")
    print(f"Learning Rate: {best_result['lr']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Best Accuracy: {best_result['accuracy']:.2f}%")
    
    return best_result

def train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.0005):
    """Simple training function"""
    
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")
    
    # Calculate class counts for information (no weights used)
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    class_counts = torch.zeros(6)
    
    # Count how many of each material we have
    for _, label in train_loader.dataset:
        class_counts[label] += 1
    
    print("\nClass distribution in training data:")
    for i, material in enumerate(materials):
        print(f"  {material}: {class_counts[i]:.0f} images")
    
    # Set up training (no class weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        
        # Testing phase
        model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(6)
        class_total = torch.zeros(6)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Count correct predictions per class
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        test_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Show detailed results for last epoch
        if epoch == epochs - 1:
            print("\nFinal results per material:")
            for i, material in enumerate(materials):
                if class_total[i] > 0:
                    acc = 100 * class_correct[i] / class_total[i]
                    print(f"  {material}: {acc:.1f}% ({class_correct[i]:.0f}/{class_total[i]:.0f})")
                else:
                    print(f"  {material}: No test samples")
            
            # Generate comprehensive evaluation
            print(f"\n" + "=" * 50)
            print("COMPREHENSIVE EVALUATION REPORT")
            print("=" * 50)
            
            # Get predictions for classification report
            all_predictions = []
            all_targets = []
            
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(all_targets, all_predictions, target_names=materials, digits=3))
            
            # Print confusion matrix info
            print(f"\nConfusion Matrix Summary:")
            print(f"Best performing material: {materials[np.argmax(class_correct/class_total)]}")
            print(f"Worst performing material: {materials[np.argmin(class_correct/class_total)]}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    return model, test_accuracy

def check_data(data_dir):
    """Simple function to check if we have data"""
    print("Checking for data...")
    
    if not os.path.exists(data_dir):
        print(f"No data found at: {data_dir}")
        return False, 0
    
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    total_images = 0
    
    for material in materials:
        folder_path = os.path.join(data_dir, material)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{material}: {count} images")
            total_images += count
    
    print(f"Total: {total_images} images")
    return True, total_images

def main():
    """Main training function"""
    print("Material Classification Training")
    print("=" * 40)
    
    # Check if we have data
    data_dir = "data"
    data_ok, total_images = check_data(data_dir)
    if not data_ok:
        print("No training data found!")
        return
    
    print(f"\nFound {total_images} images total")
    
    # Load all the data
    all_images, all_labels, total = load_data(data_dir)
    
    # Split into train and test using balanced sampling (equal samples per material)
    train_images, train_labels, test_images, test_labels = create_train_test_split(all_images, all_labels)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_images, train_labels, test_images, test_labels)
    
    # Run systematic hyperparameter testing (fast version)
    best_params = test_hyperparameters(train_images, train_labels, test_images, test_labels, epochs=3)
    
    # Train final model with best parameters - use full 10 epochs
    print(f"\n" + "=" * 60)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("=" * 60)
    
    # Recreate data loaders with best batch size
    train_loader, test_loader = create_data_loaders(
        train_images, train_labels, test_images, test_labels, 
        batch_size=best_params['batch_size']
    )
    
    # Create and train final model
    model = MaterialNet(num_classes=6)
    model_info = get_model_info(model)
    print(f"Model parameters: {model_info['total_parameters']:,}")
    
    trained_model, final_accuracy = train_model(
        model, train_loader, test_loader, 
        epochs=10, learning_rate=best_params['lr']
    )
    
    # Save the trained model
    print("\nSaving model...")
    os.makedirs("models_trained", exist_ok=True)
    
    metadata = {
        "total_images": total,
        "train_images": len(train_images),
        "test_images": len(test_images),
        "final_accuracy": final_accuracy,
        "epochs": 10,
        "best_lr": best_params['lr'],
        "best_batch_size": best_params['batch_size']
    }
    
    save_model(trained_model, "models_trained/material_classifier.pth", metadata)
    
    print("\n" + "=" * 40)
    print("Training Complete!")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    print("Model saved as: material_classifier.pth")
    print(f"Best learning rate: {best_params['lr']}")
    print(f"Best batch size: {best_params['batch_size']}")
    print("=" * 40)

if __name__ == "__main__":
    main()
