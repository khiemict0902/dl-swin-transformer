import torch
import torch.nn.functional as F
from torchvision import transforms,models
from PIL import Image
import torch.nn as nn

model = models.swin_t()
model.head = nn.Linear(model.head.in_features, 10)
print(model)
model.load_state_dict(torch.load("./out.pth", map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("./2750/River/River_1.jpg")
img = preprocess(img)
img = img.unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = model(img)

labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway','Industrial','Pasture','PẻmanentCrop','Residential','River','SeaLake']

out = output

_, index = torch.max(out, 1)
 
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
 
_, indices = torch.sort(out, descending=True)
classify = [(labels[idx], percentage[idx].item()) for idx in indices[0][:10]]

for label, perc in classify:
    print(f"{label}: {perc:.2f}%"),

