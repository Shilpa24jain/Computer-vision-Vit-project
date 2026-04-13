# Computer-vision-Vit-project
Medicinal plant Identification using Vit SSL DINO and Explainable Ai
 STEP 0: INSTALL REQUIREMENTS (RUN IN TERMINAL)
# pip install kaggle torch torchvision timm scikit-learn pandas matplotlib seaborn opencv-python pillow
# =========================================================

import os
import zipfile
import shutil
import torch
import timm
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# =========================================================
# 🔥 STEP 1: DOWNLOAD DATASET FROM KAGGLE
# =========================================================

def download_dataset():
    print("Downloading dataset from Kaggle...")

    os.system("kaggle datasets download -d sharvan123/medicinal-plant")
    os.system("kaggle datasets download -d warcoder/indian-medicinal-plant-image-dataset")

    with zipfile.ZipFile("medicinal-plant.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset1")

    with zipfile.ZipFile("indian-medicinal-plant-image-dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset2")

    print("Download & Extraction Complete!")

# =========================================================
# 🔥 STEP 2: MERGE DATASETS
# =========================================================

def merge_dataset():
    merged_path = "merged_dataset"

    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)

    os.makedirs(merged_path)

    def merge(source):
        for root, _, files in os.walk(source):
            images = [f for f in files if f.lower().endswith(('.jpg','.png','.jpeg'))]
            if len(images) > 10:
                class_name = os.path.basename(root)
                target = os.path.join(merged_path, class_name)
                os.makedirs(target, exist_ok=True)
                for img in images:
                    shutil.copy(os.path.join(root,img), target)

    merge("dataset1")
    merge("dataset2")

    print("Datasets merged successfully!")

# =========================================================
# 🔥 STEP 3: CREATE CSV DATA
# =========================================================

def create_csv():
    columns = ["ClassLabel","Plant","ScientificName","PartUsed","Uses","Dosage","Preparation","Avoid","UseType","Toxicity"]

    data = [
        [0,"Aloe vera","Aloe barbadensis","Leaf gel","Skin burns;Constipation","20-30ml juice","Extract gel","Pregnant women","Internal/External","Safe"],
        [1,"Amla","Phyllanthus emblica","Fruit","Immunity","20ml","Juice","Low BP","Internal","Safe"],
        [2,"Neem","Azadirachta indica","Leaf","Skin;Diabetes","10ml","Juice","Pregnancy","Internal/External","Moderate"],
        [3,"Tulsi","Ocimum sanctum","Leaf","Cold;Immunity","5 leaves","Raw","None","Internal","Safe"],
    ]

    df = pd.DataFrame(data, columns=columns)

    df["SafetyScore"] = df["Toxicity"].map({
        "Safe":5,"Moderate":3,"Harmful":2,"Toxic":1
    })

    df.to_csv("updated_medicinal_dataset.csv", index=False)
    print("CSV created!")

# =========================================================
# 🔥 STEP 4: TRAIN MODEL
# =========================================================

def train_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder("merged_dataset", transform=transform)

    class_names = dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    model = timm.create_model(
        "vit_base_patch16_224_dino",
        pretrained=True,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    best_val = 0

    for epoch in range(3):

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs,1)
                total += labels.size(0)
                correct += (preds==labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Accuracy: {acc:.2f}%")

        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), "model.pth")

    print("Training complete!")

# =========================================================
# 🔥 STEP 5: PREDICT
# =========================================================

def predict(img_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder("merged_dataset")
    class_names = dataset.classes

    model = timm.create_model(
        "vit_base_patch16_224_dino",
        pretrained=False,
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    pred = torch.argmax(output)

    print(f"Prediction: {class_names[pred]}")

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================

if _name_ == "_main_":
    
    # STEP 1
    download_dataset()
    
    # STEP 2
    merge_dataset()
    
    # STEP 3
    create_csv()
    
    # STEP 4
    train_model()

    # STEP 5 (example)
    # predict("test.jpg")
