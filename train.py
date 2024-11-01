import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import os
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import FileLink # just for kaggle download

class GBCUDataset(Dataset):
    def __init__(self, txt_file, img_dir, roi_file, bbox_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self.roi_data = json.load(open(roi_file))
        self.bbox_data = json.load(open(bbox_file))
        
        with open(txt_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split(',')
                self.data.append((img_name, int(label)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if img_name in self.roi_data and 'Boxes' in self.roi_data[img_name] and self.roi_data[img_name]['Boxes']:
            roi = self.roi_data[img_name]['Boxes'][0]
            image = image.crop((roi[0], roi[1], roi[2], roi[3]))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_data(batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = GBCUDataset('/kaggle/input/gbcu-data/GBCU-Shared/train.txt', 
                                '/kaggle/input/gbcu-data/GBCU-Shared/imgs', 
                                r"/kaggle/input/gbcu-data/GBCU-Shared/roi_pred.json", 
                                r"/kaggle/input/gbcu-data/GBCU-Shared/bbox_annot.json", 
                                transform=train_transform)
    test_dataset = GBCUDataset('/kaggle/input/gbcu-data/GBCU-Shared/test.txt', 
                               '/kaggle/input/gbcu-data/GBCU-Shared/imgs', 
                               r"/kaggle/input/gbcu-data/GBCU-Shared/roi_pred.json", 
                               r"/kaggle/input/gbcu-data/GBCU-Shared/bbox_annot.json", 
                               transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(ModifiedVGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ModifiedVGG19(nn.Module):
    def __init__(self, num_classes=3):
        super(ModifiedVGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, train_loader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc

def validate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    return test_loss, test_acc

def main_vgg16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = prepare_data()
    
    model = ModifiedVGG16(num_classes=3).to(device)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    num_epochs = 50
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'vgg16_best_model.pth')
    
    model.load_state_dict(torch.load('vgg16_best_model.pth'))
    final_test_loss, final_test_acc = validate(model, test_loader, criterion, device)
    print("Final Test Results:")
    print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}")
    FileLink('vgg16_best_model.pth')

    
def main_vgg19():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = prepare_data()
    
    model = ModifiedVGG19(num_classes=3).to(device)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    num_epochs = 50
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'vgg19_best_model.pth')
    
    model.load_state_dict(torch.load('vgg19_best_model.pth'))
    final_test_loss, final_test_acc = validate(model, test_loader, criterion, device)
    print("Final Test Results:")
    print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}")
    FileLink('vgg19_best_model.pth')

if __name__ == "__main__":
    main_vgg16()
    main_vgg19()
