import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

data_res = ["img\\02_070.png"]


# предобработка изображений
def preprocess_image(data_res):
    res_data_names = []
    res_data = []
    SIZE = 224

    for im in data_res:
        res_data_names.append(im)

        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        res_data.append([img])

    res_data = np.array(res_data) / 255.0

    return res_data.reshape(-1, SIZE, SIZE, 3), res_data_names


# с помощью предварительно обуч. модели InceptionV3(weights='imagenet')
# получаем признаки для входящего изображения
def intermediate_layer_model(res_data):
    model_loaded = keras.models.load_model("my_new_modelV3.h5")
    intermediate_output_res = model_loaded.predict(res_data)
    return intermediate_output_res


def dataExt(intermediate_output_res, file_res_list):
    file_res_list = [i for i in file_res_list]
    feat_res_np = intermediate_output_res
    return file_res_list, feat_res_np


# с помощью предварительно обученной модели PolySVM 
# классифицируем при помощи признаков, полученных от предыдущей модели подпись
def pred(intermediate_output_res, file_res_list):
    file_list, features = dataExt(intermediate_output_res, file_res_list)
    new_data_res = []
    # for j in file_list:
    #     index = file_list.index(j)
    new_list = list(features[0])
    new_list.extend(features[1])
    # new_list2 = list(features[1])
    # new_list2.extend(features[2])
    new_data_res.append(new_list)
    # new_data_res.append(new_list2)

    # Загрузка модели
    with open('POLY_SVM_new.pkl', 'rb') as f:
        mod = np.load(f, allow_pickle=True)
    y_pred = mod.predict(new_data_res)
    print(f"y_pred=={y_pred}")
    return y_pred


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение с подписью для верификации')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# Web приложение на стримлит
#  логотип и название
col1, col2 = st.columns([1, 1])
with col1:
    st.header('Верификация подписи с использованием ИИ')

with col2:
    st.image("img\\example.jpg")
st.title('Верификация изображений подписи в облаке Streamlit на основе предварительно обученной модели')

# Боковая панель
st.sidebar.image("img\\example.jpg", width=100)
st.sidebar.title("О проекте:")
st.sidebar.info(
    """
    Данное приложение определяет подлинность подписи. Модель обучена на 2516 экземплярах подписей.
    """
)

st.sidebar.info(
    """
    Вы можете воспользоваться данным приложением для определения подлинности подписи
    """
)

# поле для загрузки изображения
img = load_image()

result = st.button('Определить подлинность подписи')
if img and result:  # если нажата кнопка
    name = "img\\new_img.png"
    img.save(name, "PNG")  # сохраняется полученное изображение

    with st.spinner('Wait for it...'):
        time.sleep(1)
        data_res.append(name)

        x, res_data_names = preprocess_image(data_res)
        intermediate_layer = intermediate_layer_model(x)
        preds = pred(intermediate_layer, res_data_names)

        st.subheader('**Результаты распознавания:**')

        if preds[0] == 1:
            st.header("Поддельная подпись")
        elif preds[0] == 0:
            st.header("Оригинальная подпись")

# streamlit run main.py
