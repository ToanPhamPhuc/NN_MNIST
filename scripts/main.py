import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import io
from config import *
from NN import SimpleNN

class MNISTDataset(Dataset):
    def __init__(self, csv_file, train=True):
        self.data = pd.read_csv(csv_file)
        if train:
            self.labels = self.data['label'].values
            self.images = self.data.drop('label', axis=1).values
        else:
            self.labels = None
            self.images = self.data.values
        self.images = self.images.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, IMAGE_SIZE, IMAGE_SIZE)
        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(image), torch.tensor(label, dtype=torch.long)
        else:
            return torch.tensor(image)

class DrawingApp:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.root = tk.Tk()
        self.root.title("Draw a Digit (0-9)")
        self.root.geometry("400x500")
        
        # Drawing canvas
        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        # Buttons
        self.predict_btn = Button(self.root, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=5)
        
        self.clear_btn = Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)
        
        # Result label
        self.result_label = Label(self.root, text="Draw a digit and click Predict", font=("Arial", 14))
        self.result_label.pack(pady=10)
        
        self.drawing = False
        self.drawn_pixels = set()  # Store drawn pixel coordinates
        
    def paint(self, event):
        self.drawing = True
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        
        # Store the drawn pixels
        for x in range(max(0, int(x1)), min(280, int(x2) + 1)):
            for y in range(max(0, int(y1)), min(280, int(y2) + 1)):
                self.drawn_pixels.add((x, y))
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawn_pixels.clear()
        self.result_label.config(text="Draw a digit and click Predict")
        
    def predict(self):
        # Create image from drawn pixels
        img = Image.new('L', (280, 280), 255)  # White background
        
        # Draw the pixels that were drawn on canvas
        for x, y in self.drawn_pixels:
            if 0 <= x < 280 and 0 <= y < 280:
                img.putpixel((x, y), 0)  # Black pixel
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert (white background to black)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Reshape for model input
        img_tensor = torch.tensor(img_array).reshape(1, 1, 28, 28)
        img_tensor = img_tensor.to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_digit = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_digit].item() * 100
            
        # Display result
        result_text = f"Predicted: {predicted_digit} (Confidence: {confidence:.2f}%)"
        self.result_label.config(text=result_text)
        
        # Show probabilities for all digits
        print(f"\nPrediction: {predicted_digit} with {confidence:.2f}% confidence")
        print("All digit probabilities:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  {i}: {prob.item()*100:.2f}%")

def train():
    # Load and split data
    full_dataset = MNISTDataset(TRAIN_CSV, train=True)
    train_idx, val_idx = train_test_split(np.arange(len(full_dataset)), test_size=0.1, random_state=42)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {acc:.4f}")
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    test_dataset = MNISTDataset(TEST_CSV, train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    # Save submission
    submission = pd.read_csv(SUBMISSION_CSV)
    submission['Label'] = predictions
    submission.to_csv('submission.csv', index=False)

def interactive_test():
    """Interactive testing with drawing interface"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print("Loading drawing interface...")
    app = DrawingApp(model, device)
    app.root.mainloop()

if __name__ == "__main__":
    choice = input("Choose mode:\n1. Train model\n2. Generate predictions\n3. Interactive test (draw digits)\nEnter choice (1/2/3): ")
    
    if choice == "1":
        train()
    elif choice == "2":
        predict()
    elif choice == "3":
        interactive_test()
    else:
        print("Invalid choice. Running full pipeline (train + predict)...")
        train()
        predict()
