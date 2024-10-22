import pandas as pd
import numpy as np
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from prophet import Prophet
import matplotlib.pyplot as plt


def price_forecasting(data, data_forecast, show=False):
    if show:
        plt.figure(figsize=(15,7))  # посмотрим на график
        plt.plot(data['y'])
        plt.show()
    result = adfuller(data['y'])  # Тест Дики-Фуллера на стационарность временного ряда
    print(f"p-value raw data: {result[1]}\n")
    # Так как p-value > 0.05, то ряд не стационарный, приведем его к стационарному, вычтем из ряда тренд и сезонность
    decomposition = seasonal_decompose(data['y'], model='additive', period=71)  # Декомпозиция временного ряда 
    trend_raw = decomposition.trend
    seasonal_raw = decomposition.seasonal
    trend = trend_raw.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')  # Заполняем пропуски
    seasonal = seasonal_raw.fillna(seasonal_raw.mean()).fillna(method='bfill').fillna(method='ffill')  # Заполняем пропуски 
    X = np.arange(len(trend)).reshape(-1, 1)
    model_trend = LinearRegression()  # Обучаем линейную регрессию для прогнозирования тренда
    model_trend.fit(X[240:], trend.values[240:])  # Обучаем линейную регрессию: пропуская начало, где визуально нет тренда
    future_X = np.arange(len(trend), len(trend) + 40).reshape(-1, 1)  
    future_trend = model_trend.predict(future_X)  # Прогнозируем тренд на будущее
    seasonal_values = seasonal.values[-365:]  # Сезонность за последний год
    model_seasonal = AutoReg(seasonal_values, lags=7).fit()  # Обучаем авторегрессионную модель для сезонности
    future_seasonal = model_seasonal.predict(start=len(seasonal_values), end=len(seasonal_values) + 40 - 1)  # Прогнозируем
    data['y_detrended'] = data['y'] - trend - seasonal
    result_detrended = adfuller(data['y_detrended']) # еще раз проверяем стал ли ряд стационарным после вычитания
    print(f"p-value detrended: {result_detrended[1]}")
    model = Prophet(daily_seasonality=False, weekly_seasonality=False)
    data_for_predict = pd.DataFrame({'ds': data['ds'].values, 'y': data['y_detrended'].values})
    model.fit(data_for_predict)  # Обучение Prophet на очищенных от тренда и сезонности данных
    future = model.make_future_dataframe(periods=40)
    forecast = model.predict(future)  # Прогнозируем с Prophet на 40 дней вперед
    forecast_future = forecast.iloc[len(trend):].copy()
    # Для спрогнозированных данных добавляем обратно отдельно спрогнозированные тренд и сезонность
    forecast_future['yhat_final'] = forecast_future['yhat'] + future_trend + future_seasonal
    if show:
        plt.figure(figsize=(10, 6))  # отрисуем прогноз на одном графике с исходными данными
        plt.plot(data['ds'], data['y'], label='Исходные данные')
        plt.plot(forecast_future['ds'], forecast_future['yhat_final'], label='Прогноз с трендом и сезонностью')
        plt.title('Прогноз с использованием Prophet и прогнозированием тренда и сезонности')
        plt.legend()
        plt.show()
    forecast_values = forecast_future['yhat_final'].round(2).tolist()
    try:  # Сохранение данных в JSON файл
        with open('forecast_value.json', 'w') as file:
            json.dump(forecast_values, file)
        print("Данные успешно сохранены в 'forecast_value.json'.")
    except Exception as e:
        print(f"Произошла ошибка при записи данных: {e}")
    
    future_dates = pd.to_datetime(data_forecast['дата'])  # Даты будущего из файла для предсказаний
    future_values = forecast_future['yhat_final']  # Прогнозируемые значения
    future_data = pd.DataFrame({
        'ds': future_dates.values,
        'y': future_values.values
    })
    if 'ds' in data.columns:
        data = data.set_index('ds')
    data = pd.concat([data, future_data.set_index('ds')])
    return data


def binary_classification(data):
    # Генерируем новые признаки
    data['lag_1'] = data['y'].shift(1)  # Цена на 1 шаг назад
    data['lag_2'] = data['y'].shift(2)  # Цена на 2 шага назад

    # Разность между текущей и предыдущей ценой
    data['diff_lag_1'] = data['y'] - data['lag_1']
    data['diff_lag_2'] = data['y'] - data['lag_2']

    # Скользящие средние
    data['ma_3'] = data['y'].rolling(window=3).mean()  # Скользящая средняя за 3 дня
    data['ma_7'] = data['y'].rolling(window=7).mean()  # Скользящая средняя за 7 дней
    data['ma_14'] = data['y'].rolling(window=14).mean()  # Скользящая средняя за 14 дней

    # Процентное изменение цены
    data['pct_change'] = data['y'].pct_change()

    data = data.fillna(method='bfill')  # Заполнение NaN значений
    X = data[['y', 'lag_1', 'lag_2', 'diff_lag_1', 'diff_lag_2', 'ma_3', 'ma_7', 'ma_14', 'pct_change']][:40]
    y = data['target'][:40]

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)

    future_pred = xgb_model.predict(X[-40:]).tolist()
    # Сохранение данных в JSON файл
    try:
        with open('forecast_class.json', 'w') as file:
            json.dump(future_pred, file)
        print("Данные успешно сохранены в 'forecast_class.json'.")
    except Exception as e:
        print(f"Произошла ошибка при записи данных: {e}")


if __name__ == "__main__":
    data = pd.read_csv("data.csv", decimal=',', index_col=['дата'], parse_dates=True, dayfirst=True)
    data = data.sort_values('дата')
    data.reset_index(inplace=True)
    data = data.rename(columns={'дата': 'ds', 'выход': 'y', 'направление': 'target'})
    data['target'] = data['target'].map({'ш': 0, 'л': 1})  # Перекодируем 'направление' ('ш' -> 0, 'л' -> 1)

    data_forecast = pd.read_csv("forecast.csv", decimal=',', index_col=['дата'], parse_dates=True, dayfirst=True)
    data_forecast = data_forecast.sort_values('дата')
    data_forecast.reset_index(inplace=True)
    data_with_forcast = price_forecasting(data=data, data_forecast=data_forecast)  # Точечный прогноз
    binary_classification(data=data_with_forcast)  # Прогноз бинарной классификации