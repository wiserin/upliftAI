import joblib
import pandas as pd

# Загрузка модели
uplift_model = joblib.load('uplift_model.pkl')

# Загрузка новых данных
test_data = pd.read_csv('test.csv')

# Удаляем колонки, которые не нужны для предсказания
X_test = test_data.drop(columns=['retro_date'], errors='ignore')  # Удаляем ненужные колонки

# Преобразование категориальных данных в dummies
X_test_dummies = pd.get_dummies(X_test, drop_first=True)

# Загрузка обучающих данных для получения колонок
train_data = pd.read_csv('train.csv')
X_train = train_data.drop(columns=['successful_utilization', 'treatment', 'retro_date'])
X_train_dummies = pd.get_dummies(X_train, drop_first=True)

# Убедимся, что колонки совпадают, добавляя недостающие колонки в X_test_dummies
X_test_final = X_test_dummies.reindex(columns=X_train_dummies.columns, fill_value=0)

# Прогнозируем uplift
uplift_preds = uplift_model.predict(X_test_final)

# Сохранение предсказаний в файл для сдачи
test_data['successful_utilization'] = uplift_preds
test_data.to_csv('uplift_predictions.csv', index_label='index')