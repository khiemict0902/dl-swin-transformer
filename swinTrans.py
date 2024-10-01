import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.swin_t()
print(model)

num_classes = 10
model.head = nn.Linear(model.head.in_features, num_classes)

# Move the model to the GPU if available
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root='./2750', transform=transform)

print(train_dataset)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()  
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            # Move images and labels to the device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            print(running_loss)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Print statistics for the epoch
        epoch_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
torch.save(model.state_dict(),'out.pth')


