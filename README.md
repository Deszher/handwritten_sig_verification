# streamlit_mlproject
_________________________________________
## Приложение по 
Web-приложение на основе библиотеки Streamlit, 
которое проверяет подлинность рукописной подписи.

Приложение с помощью предварительно обученной модели InceptionV3(weights='imagenet')
извлекает признаки для входящего изображения с подписью.

Далее с использованием модели PolySVM
классифицируем при помощи признаков, полученных от предыдущей модели, подпись

## Запуск WEB-приложения streamlit

### Использование без докера
Сначала нужно установить anaconda https://docs.anaconda.com/free/anaconda/install/mac-os/
```bash
# Без venv
# Устанавливаем зависимости
pip install -r requirements.txt
# Запускаем WEB-приложение
streamlit run main.py

# С venv
# Создаем venv
python3 -m venv .venv
source .venv/bin/activate
# Устанавливаем зависимости в venv
pip install -r requirements.txt
# Запускаем WEB-приложение
streamlit run main.py
```

### Использование с докером
```bash
# Собираем образ
docker build -t handwritten_sig_verification

# Запускаем
docker run -p 8501:8501 handwritten_sig_verification
```

 Веб сервер будет доступен по адресу http://localhost:8501/