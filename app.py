import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import CatDogCNN

# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶")

st.title("🐱 Cat vs Dog Image Classifier")
st.write("Upload an image and the model will predict whether it is a **Cat** or **Dog**.")

# Load model
@st.cache_resource
def load_model():
    model = CatDogCNN()
    model.load_state_dict(torch.load("cat_dog_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# File uploader
uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, 1).item()

        if prediction == 0:
            st.success("🐱 This is a CAT")
        else:
            st.success("🐶 This is a DOG")
