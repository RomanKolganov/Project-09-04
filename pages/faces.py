import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import cv2

# Функция для загрузки модели
weights_path = 'models/best_faces.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
model.to(device).eval()

st.title("Детекция и маскировка лиц 🎭")
st.markdown("Загрузите изображения или вставьте прямую ссылку на изображение.")

# Загрузка файлов пользователем
uploaded_files = st.file_uploader("Выберите изображения...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Загрузка файлов по прямым ссылкам
urls = st.text_area("Или вставьте ссылки на изображения (каждую ссылку с новой строки):", height=150)

# Установка порога для отрисовки рамок
# threshold = st.slider("Установите порог отрисовки рамок:", 0.0, 1.0, 0.5)
threshold = 0.5

images = []

# Обработка загруженных файлов
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append(image)

# Обработка загруженных по ссылкам изображений
if urls:
    try:
        url_list = urls.splitlines()
        for url in url_list:
            response = requests.get(url.strip())
            image = Image.open(BytesIO(response.content))
            images.append(image)
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения по ссылке: {e}")

# Обработка и отображение изображений
for i, image in enumerate(images):  # Используем enumerate для получения индекса

    # Преобразуем изображение для модели и выполняем предсказание
    img_np = np.array(image)

    with torch.no_grad():
        results = model(img_np)

    detections = results.xyxy[0]

    for *bbox, conf, cls in detections:
        if conf >= threshold:
            x1, y1, x2, y2 = map(int, bbox)

            # Маскирование области со стремлением (блюр)
            mask = img_np[y1:y2, x1:x2]
            blur_mask = cv2.GaussianBlur(mask, (89, 89), 0)
            img_np[y1:y2, x1:x2] = blur_mask

    # Отображение результата
    st.image(img_np, caption="Изображение с размытой детектированной областью", use_column_width=True)

    # Сохранение результата для скачивания
    output_image_path = f'output_image_{i}.png'  # Уникальное имя файла для каждого изображения
    Image.fromarray(img_np).save(output_image_path)

    # Кнопка для скачивания результата с уникальным ключом
    with open(output_image_path, "rb") as file:
        st.download_button(label="Скачать итоговое изображение", data=file, file_name=f"output_image_{i}.png", mime="image/png", key=f"download_button_{i}")  # Используем уникальный ключ
st.write('')
st.write('')
st.write('')
st.markdown("Сервис реализован с использованием YOLOv5:")
st.write('- 5 эпох обучения')
st.write('- 16.7 тысяч картинок в датасете')
st.image('Images/faces_metrics/PR_curve_faces.png', use_column_width=True)
st.image('Images/faces_metrics/confusion_matrix_faces.png', use_column_width=True)
st.image('Images/faces_metrics/results_faces.png', use_column_width=True)