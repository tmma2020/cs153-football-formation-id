# resnet_processed.py

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report

# Paths
RAW_DATA = "/cs/cs153/data/toms_project_data/hudl_dataset"
DATASET_JSON = "/cs/cs153/data/toms_project_data/hudl_dataset/dataset.json"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HudlProcessedDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = [json.loads(line) for line in f]

        # Map each cropped image to its label
        self.data = []
        for entry in metadata:
            formation = entry["off_formation"]
            video_path = entry["video_path"]
            image_filename = f"cropped_sideline_{video_path}.png"

            for formation_folder in os.listdir(root_dir):
                formation_path = os.path.join(root_dir, formation_folder)
                if not os.path.isdir(formation_path):
                    continue

                for video_folder in os.listdir(formation_path):
                    video_path_folder = os.path.join(formation_path, video_folder)
                    if not os.path.isdir(video_path_folder):
                        continue

                    if os.path.exists(os.path.join(video_path_folder, image_filename)):
                        full_path = os.path.join(video_path_folder, image_filename)
                        self.data.append((full_path, formation))
                        break

        # Label encoding
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(x[1] for x in self.data)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_to_idx[label]
        return image, label_idx

def run_resnet_processed():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = HudlProcessedDataset(RAW_DATA, DATASET_JSON, transform=transform)
    num_classes = len(dataset.label_to_idx)

    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pretrained ResNet
    model = models.resnet18(pretrained=True)

    # Modify final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Compute accuracy
    correct = sum(np.array(all_labels) == np.array(all_preds))
    total = len(all_labels)
    print(f"Validation Accuracy on Processed Images: {100 * correct / total:.2f}%")

    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=list(dataset.label_to_idx.keys()))
    output_dir = "outputs_resnet"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "classification_report_processed.txt"), "w") as f:
        f.write(report)

    print(f"[INFO] Classification report saved to {os.path.join(output_dir, 'classification_report_processed.txt')}")


    # Save trained model
    model_save_path = os.path.join(output_dir, "resnet_processed_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Trained ResNet model saved to {model_save_path}")

if __name__ == "__main__":
    run_resnet_processed()
