import streamlit as st
import torch
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# Загрузка предобученной модели YOLOv5
@st.cache_resource
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

# Пример пути к весам
weights_path = "models/ships_weights.pt"
model = load_model(weights_path)

# Устанавливаем минимальную вероятность для детекций
model.conf = 0.3

# Функция для преобразования изображений
def process_image(image):
    return image.convert("RGB")  # Убедитесь, что изображение в формате RGB

# Функция для детекции объектов
def detect_objects(images):
    results = model(images)  # Модель принимает список изображений
    return results

# Информация о модели
epochs = 100
sample_size = 26881
metrics = {
    'PR_curve': 'Images/ship_metrics/PR_curve.png',
    'Confusion_matrix': 'Images/ship_metrics/confusion_matrix.png',
    'P_curve': 'Images/ship_metrics/P_curve.png',
    'F1_curve': 'Images/ship_metrics/F1_curve.png',
    'R_curve': 'Images/ship_metrics/R_curve.png',
    'Сводная таблица': 'Images/ship_metrics/results.png'
}

# Раздел для загрузки изображений
st.title("Детекция кораблей с использованием YOLOv5")
st.subheader("Загрузите изображения для детекции")

# Поддержка загрузки нескольких файлов
uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Чтение и отображение изображения
        img = Image.open(uploaded_file).convert("RGB")

        # Выполнение детекции
        model.eval()
        with torch.no_grad():
            results = model(img)

        # Отображение результатов
        st.header(f"Результаты детекции")
        annotated_image = results.render()[0]  # Получение изображения с аннотациями
        st.image(annotated_image, caption="Обнаруженные объекты", use_column_width=True)


# Раздел для загрузки по прямой ссылке
st.subheader("Загрузите изображение по URL")
url = st.text_input('Введите URL изображения')
if st.button('Загрузить изображение'):
    if url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img_url = Image.open(BytesIO(response.content)).convert("RGB")

                # Выполнение детекции
                model.eval()
                with torch.no_grad():
                    results = model(img_url)

                # Отображение результатов
                st.header("Результаты детекции из URL:")
                annotated_image = results.render()[0]  # Получение изображения с аннотациями
                st.image(annotated_image, caption="Обнаруженные объекты", use_column_width=True)
            else:
                st.error('Не удалось загрузить изображение. Пожалуйста, проверьте URL или попробуйте позже.')
        except Exception as e:
            st.error(f'Ошибка загрузки изображения: {e}')

# Информация о модели
st.subheader("Информация о модели")
st.write("Количество эпох обучения:", epochs)
st.write("Объем выборки:", sample_size)

# Отображение графиков PR и confusion matrix
if os.path.exists(metrics['PR_curve']):
    st.image(metrics['PR_curve'], caption="PR Кривая")
else:
    st.warning("График PR не найден.")

if os.path.exists(metrics['Confusion_matrix']):
    st.image(metrics['Confusion_matrix'], caption="Матрица ошибок")
else:
    st.warning("Матрица ошибок не найдена.")

if os.path.exists(metrics['P_curve']):
    st.image(metrics['P_curve'], caption="Precision кривая")
else:
    st.warning("График P не найден.")

if os.path.exists(metrics['F1_curve']):
    st.image(metrics['F1_curve'], caption="F1 кривая")
else:
    st.warning("График F1 не найден.")

if os.path.exists(metrics['R_curve']):
    st.image(metrics['R_curve'], caption="R кривая")
else:
    st.warning("График R не найден.")

if os.path.exists(metrics['Сводная таблица']):
    st.image(metrics['Сводная таблица'], caption="Сводная таблицак")
else:
    st.warning("Сводная таблица не найдена.")