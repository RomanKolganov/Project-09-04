
import streamlit as st
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import requests
from io import BytesIO

# Определение класса UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Энкодер
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)

        # Декодер
        u1 = self.up1(c3)
        u1 = torch.cat([u1, c2], dim=1)
        u1 = self.conv4(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, c1], dim=1)
        u2 = self.conv5(u2)

        return torch.sigmoid(self.final(u2))

# Заголовок страницы
st.title("U-Net Semantic Segmentation with PyTorch")

# Функция для загрузки модели
@st.cache_resource
def load_unet_model(model_path="models/unet_model.pth"):
    model = UNet()  # Создаем экземпляр модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Проверяем доступность CUDA
    model.load_state_dict(torch.load(model_path, map_location=device))  # Загружаем веса модели на правильное устройство
    model.eval()  # Переводим модель в режим оценки
    return model

# Загрузка предобученной модели
model = load_unet_model()

# Загрузка изображений
uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_url = st.text_input("Or enter an image URL")

# Предсказания
if uploaded_images or image_url:
    images_to_predict = []

    # Если есть загруженные изображения
    if uploaded_images:
        for img_file in uploaded_images:
            img = Image.open(img_file).convert("RGB")
            images_to_predict.append(img)

    # Если указана ссылка на изображение
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Проверка на успешный запрос
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images_to_predict.append(img)
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

    # Функция для предобработки изображений
    def preprocess_images(images):
        processed_images = []
        for img in images:
            img = img.resize((128, 128))  # Измените размер на нужный
            img = transforms.ToTensor()(img)  # Конвертация в тензор
            processed_images.append(img)
        return torch.stack(processed_images)  # Возвращаем батч изображений

    # Предобработка изображений
    X = preprocess_images(images_to_predict)

    # Предсказание масок с использованием загруженной модели
    with torch.no_grad():
        predictions = model(X)  # Модель возвращает предсказания
        predictions = (predictions > 0.5).int()  # Приведение к бинарной маске

    # Отображение предсказанных масок
    st.subheader("Predictions")
    for i in range(len(images_to_predict)):
        st.write(f"Image {i + 1}")
        st.image(np.asarray(X[i].permute(1, 2, 0) * 255, dtype=np.uint8), caption="Original Image", use_column_width=True)
        st.image(predictions[i][0].numpy() * 255, caption="Predicted Mask", use_column_width=True)  # Например, берем первую канал маски


st.write('Модель - UNet')
st.write('Количество эпох - 30')
st.write('Размер Выборки - 5108')


image_path1 = "Images/aero_images/metrics.jpg"  # Замените на путь к вашему изображению
image_path2 = "Images/aero_images/metrics_2.jpg"  # Замените на путь к вашему изображению
image_path3 = "Images/aero_images/confusion_matrix.jpg"  # Замените на путь к вашему изображению

# Загрузка и отображение первого изображения
img1 = Image.open(image_path1).convert("RGB")
st.image(img1, caption="metrics" , use_column_width=True)

# Загрузка и отображение второго изображения
img2 = Image.open(image_path2).convert("RGB")
st.image(img2, caption="metrics 2.", use_column_width=True)

# Загрузка и отображение третьего изображения
img3 = Image.open(image_path3).convert("RGB")
st.image(img3, caption="confusion matrix", use_column_width=True)
    
