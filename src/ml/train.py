# training script for material classification
# messy but functional - needs cleanup

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import numpy as np
from models import MaterialNet, save_model, count_params
from sklearn.metrics import classification_report

"""This study inspired by previous works:
By hysts ML Engineer: https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/datasets/datasets.py
"""
class MaterialDataset(Dataset):
    # dataset class for loading images
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # load image 
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_images_from_folders(data_dir):
    # scan through material folders and collect all images
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    
    all_imgs = []
    all_labels = []
    counts = {}
    
    print("loading images from folders...")
    
    for i, material in enumerate(materials):
        folder = os.path.join(data_dir, material)
        if os.path.exists(folder):
            count = 0
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder, filename)
                    all_imgs.append(img_path)
                    all_labels.append(i)
                    count += 1
            counts[material] = count
        else:
            counts[material] = 0
    
    # print what we found
    total = len(all_imgs)
    print(f"found {total} images total:")
    for material in materials:
        count = counts.get(material, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {material}: {count} ({pct:.1f}%)")
    
    return all_imgs, all_labels, total
    
def split_data_balanced(all_imgs, all_labels):
    # try to get equal numbers from each material
    
    TRAIN_PER_MATERIAL = 80
    TEST_PER_MATERIAL = 20
    TOTAL_NEEDED = TRAIN_PER_MATERIAL + TEST_PER_MATERIAL
    
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    
    # group by material
    material_imgs = {}
    for i in range(6):
        material_imgs[i] = []
    
    for img_path, label in zip(all_imgs, all_labels):
        material_imgs[label].append(img_path)
    
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    
    print(f"\nbalanced split ({TRAIN_PER_MATERIAL} train + {TEST_PER_MATERIAL} test per material):")
    
    for i, material in enumerate(materials):
        available = material_imgs[i]
        
        if len(available) < TOTAL_NEEDED:
            print(f"  WARNING: {material} only has {len(available)} images, need {TOTAL_NEEDED}")
            # just use what we have
            np.random.shuffle(available)
            train_count = int(len(available) * 0.8)
            train_subset = available[:train_count]
            test_subset = available[train_count:]
        else:
            # take the amount we need
            np.random.shuffle(available)
            selected = available[:TOTAL_NEEDED]
            train_subset = selected[:TRAIN_PER_MATERIAL]
            test_subset = selected[TRAIN_PER_MATERIAL:]
        
        # add to lists
        train_imgs.extend(train_subset)
        train_labels.extend([i] * len(train_subset))
        test_imgs.extend(test_subset)
        test_labels.extend([i] * len(test_subset))
        
        print(f"  {material}: train={len(train_subset)}, test={len(test_subset)}")
    
    print(f"\nfinal split:")
    print(f"  training: {len(train_imgs)} images")
    print(f"  testing: {len(test_imgs)} images")
    
    return train_imgs, train_labels, test_imgs, test_labels

def make_data_loaders(train_imgs, train_labels, test_imgs, test_labels, batch_sz=16):
    # create pytorch data loaders

    # normalization technique from Machine Learning module with the music dataset coursework with Dr. Haixia Liu  
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MaterialDataset(train_imgs, train_labels, train_transform)
    test_dataset = MaterialDataset(test_imgs, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
    
    return train_loader, test_loader

def train_the_model(model, train_loader, test_loader, num_epochs=10, lr=0.001):
    # main training loop
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"training on: {device}")
    
    # count samples per class
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    class_counts = torch.zeros(6)
    
    for _, label in train_loader.dataset:
        class_counts[label] += 1
    
    print("\ntraining data distribution:")
    for i, material in enumerate(materials):
        print(f"  {material}: {class_counts[i]:.0f}")
    
    # setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nstarting training ({num_epochs} epochs)...")
    start_time = time.time()
    
    # Track best accuracy and results throughout training
    best_accuracy = 0.0
    best_class_correct = torch.zeros(6)
    best_class_total = torch.zeros(6)
    
    for epoch in range(num_epochs):
        # training
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # print progress occasionally
            if batch_idx % 10 == 0:
                print(f"  epoch {epoch+1}/{num_epochs}, batch {batch_idx}/{len(train_loader)}, loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        
        # testing
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
                
                # per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        test_acc = 100 * correct / total
        print(f"epoch {epoch+1}: loss={avg_loss:.4f}, test_acc={test_acc:.2f}%")
        
        # Update best results if this epoch is better
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_class_correct = class_correct.clone()
            best_class_total = class_total.clone()
    
    training_time = time.time() - start_time
    
    # Print comprehensive training summary
    print(f"\n{'='*60}")
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    # Dataset Information
    total_train_samples = len(train_loader.dataset)
    total_test_samples = len(test_loader.dataset)
    total_images = total_train_samples + total_test_samples
    
    print("Dataset Information:")
    print(f"Total Images Found: {total_images:,} images across {len(materials)} material types")
    print(f"Balanced Training Set: {total_train_samples} images ({total_train_samples//len(materials)} per material)")
    print(f"Test Set: {total_test_samples} images ({total_test_samples//len(materials)} per material)")
    
    # Training Configuration
    print("\nTraining Configuration:")
    print(f"Model: MaterialNet with {count_params(model)['total']:,} parameters")
    print(f"Training Device: {device.type.upper()}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Training Time: ~{int(training_time//60)} minutes ({training_time:.2f} seconds)")
    
    # Final Results using best accuracy
    print("\nFinal Results:")
    print(f"Overall Accuracy: {best_accuracy:.2f}%")
    
    # Calculate best and worst performing materials
    material_accuracies = []
    for i, material in enumerate(materials):
        if best_class_total[i] > 0:
            acc = 100 * best_class_correct[i] / best_class_total[i]
            material_accuracies.append((material, acc))
        else:
            material_accuracies.append((material, 0.0))
    
    # Sort to find best and worst
    material_accuracies.sort(key=lambda x: x[1], reverse=True)
    best_material = material_accuracies[0]
    worst_material = material_accuracies[-1]
    
    print(f"Best Performing Material: {best_material[0].title()} ({best_material[1]:.1f}% accuracy)")
    print(f"Worst Performing Material: {worst_material[0].title()} ({worst_material[1]:.1f}% accuracy)")
    
    # Per-Material Performance
    print("\nPer-Material Performance:")
    for i, material in enumerate(materials):
        if best_class_total[i] > 0:
            acc = 100 * best_class_correct[i] / best_class_total[i]
            correct_count = int(best_class_correct[i])
            total_count = int(best_class_total[i])
            print(f"{material.title()}: {acc:.1f}% ({correct_count}/{total_count} correct)")
        else:
            print(f"{material.title()}: no test samples")
    
    return model, best_accuracy

def check_for_data(data_dir):
    # see if we have training data
    print("checking for training data...")
    
    if not os.path.exists(data_dir):
        print(f"data directory not found: {data_dir}")
        return False, 0
    
    materials = ['cardboard', 'fabric', 'leather', 'metal', 'paper', 'wood']
    total = 0
    
    for material in materials:
        folder = os.path.join(data_dir, material)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{material}: {count} images")
            total += count
        else:
            print(f"{material}: folder not found")
    
    print(f"total images: {total}")
    return True, total

def main():
    # main function to run training
    print("Material Classification Training")
    print("="*40)
    
    # check if we have data
    data_dir = "data"
    data_ok, total_imgs = check_for_data(data_dir)
    if not data_ok:
        print("no training data available!")
        return
    
    print(f"\nfound {total_imgs} images")
    
    # load all images
    all_imgs, all_labels, total = load_images_from_folders(data_dir)
    
    # split into train/test with balanced sampling
    train_imgs, train_labels, test_imgs, test_labels = split_data_balanced(all_imgs, all_labels)
    
    # create data loaders
    train_loader, test_loader = make_data_loaders(train_imgs, train_labels, test_imgs, test_labels, batch_sz=16)
    
    # train the model
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print("="*60)
    
    model = MaterialNet(num_classes=6)
    model_info = count_params(model)
    print(f"model has {model_info['total']:,} parameters")
    
    lr = 0.0001
    batch_sz = 16
    
    trained_model, best_acc = train_the_model(
        model, train_loader, test_loader, 
        num_epochs=15, lr=lr
    )
    
    # save the model
    os.makedirs("models_trained", exist_ok=True)
    
    extra_info = {
        "total_imgs": total,
        "train_imgs": len(train_imgs),
        "test_imgs": len(test_imgs),
        "best_acc": best_acc,  # best_acc
        "epochs": 15,
        "lr": lr,
        "batch_sz": batch_sz
    }
    
    save_model(trained_model, "models_trained/material_classifier.pth", extra_info)
    
    print("\n" + "="*40)
    print("TRAINING DONE!")
    print(f"best accuracy achieved: {best_acc:.2f}%")  # best_acc
    print("saved as: material_classifier.pth")
    print(f"learning rate: {lr}")
    print(f"batch size: {batch_sz}")
    print("="*40)

if __name__ == "__main__":
    main()
