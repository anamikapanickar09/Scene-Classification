import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Class labels
class_names = ['confined', 'normal', 'open']
num_classes = len(class_names)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_scene_classifier.pth", map_location=device))
model.to(device)
model.eval()


# Image paths
images = [
    ("confined", "./training_data/valid/confined_space/04221.jpg"),
    ("normal", "./training_data/valid/normal_space/00115.jpg"),
    ("open", "./training_data/valid/open_space/00259.jpg"),
]

for scene_class, img_path in images:
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = class_names[pred.item()]

    print(f"{img_path}\t: {label} space, prediction {label == scene_class}")
