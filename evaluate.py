import torch
import torch.nn.functional as F
from torchvision import transforms,models,datasets
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.swin_t()
model.head = nn.Linear(model.head.in_features, 10)
# model.to(device)
model.load_state_dict(torch.load("./model.pth", map_location="cpu"))

model.eval()

num_classes =10


transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='./2750', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)

true_positive = torch.zeros(num_classes)
false_positive = torch.zeros(num_classes)
false_negative = torch.zeros(num_classes)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Make predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
        
        # Update accuracy count
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # For precision/recall per class
        for i in range(num_classes):
            true_positive[i] += ((preds == i) & (labels == i)).sum().item()
            false_positive[i] += ((preds == i) & (labels != i)).sum().item()
            false_negative[i] += ((preds != i) & (labels == i)).sum().item()

# Calculate Accuracy
accuracy = correct / total

# Calculate Precision, Recall, and F1-Score per class
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * precision * recall / (precision + recall)

# Macro averages
precision_mean = precision.mean().item()
recall_mean = recall.mean().item()
f1_mean = f1.mean().item()


# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision_mean}')
print(f'Recall: {recall_mean}')
print(f'F1-Score: {f1_mean}')
