from torchvision import datasets

dataset = datasets.ImageFolder("dataset/train")

print("Classes:", dataset.classes)
print("Total images:", len(dataset))
