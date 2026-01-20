import torch
from torchvision import transforms
from PIL import Image
from model import CatDogCNN

model = CatDogCNN()
model.load_state_dict(torch.load("cat_dog_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open("test.jpg")
image = transform(image).unsqueeze(0)

output = model(image)
prediction = torch.argmax(output, 1).item()

if prediction == 0:
    print("Predicted: Cat 🐱")
else:
    print("Predicted: Dog 🐶")
