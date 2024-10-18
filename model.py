import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklift.models import TwoModels
from sklearn.ensemble import RandomForestClassifier
from sklift.metrics import uplift_at_k
from sklearn.feature_selection import SelectFromModel

# Загрузка данных
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Отделяем целевые переменные
X = train_data.drop(columns=['successful_utilization', 'treatment', 'retro_date'])  # Удаляем целевую переменную и дату
y = train_data['successful_utilization']
treatment = train_data['treatment']

# Преобразование категориальных данных
X = pd.get_dummies(X)
X_test = test_data.drop(columns=['retro_date'], errors='ignore')  # Удаляем ненужные колонки
X_test = pd.get_dummies(X_test)

# Убедимся, что колонки совпадают
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Разделение данных на тренировочные и тестовые выборки (на небольшую подвыборку для поиска гиперпараметров)
X_train, X_val, y_train, y_val, treatment_train, treatment_val = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42)

# Уменьшение тренировочной выборки для поиска гиперпараметров
X_train_subsample, _, y_train_subsample, _, treatment_train_subsample, _ = train_test_split(
    X_train, y_train, treatment_train, test_size=0.9, random_state=42)  # Используем 10% данных

# Определение гиперпараметров для RandomizedSearchCV для каждой из моделей
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Настройка модели для группы лечения
model_trmnt = RandomForestClassifier(random_state=42)
random_search_trmnt = RandomizedSearchCV(
    model_trmnt,
    param_distributions=param_grid,
    n_iter=20,  # Увеличили количество итераций
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search_trmnt.fit(X_train_subsample[treatment_train_subsample == 1], y_train_subsample[treatment_train_subsample == 1])

# Настройка модели для контрольной группы
model_ctrl = RandomForestClassifier(random_state=42)
random_search_ctrl = RandomizedSearchCV(
    model_ctrl,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search_ctrl.fit(X_train_subsample[treatment_train_subsample == 0], y_train_subsample[treatment_train_subsample == 0])

# Применение лучших параметров для обеих моделей в TwoModels
best_model_trmnt = random_search_trmnt.best_estimator_
best_model_ctrl = random_search_ctrl.best_estimator_

# Отбор признаков с помощью feature importance
feature_selector_trmnt = SelectFromModel(best_model_trmnt, threshold='mean', prefit=True)
feature_selector_ctrl = SelectFromModel(best_model_ctrl, threshold='mean', prefit=True)

X_train_selected = feature_selector_trmnt.transform(X_train)
X_val_selected = feature_selector_trmnt.transform(X_val)
X_test_selected = feature_selector_trmnt.transform(X_test)

# Применение к модели TwoModels с отобранными признаками
uplift_model = TwoModels(
    estimator_trmnt=best_model_trmnt,
    estimator_ctrl=best_model_ctrl
)

# Обучение полной модели на отобранных признаках
uplift_model.fit(X_train_selected, y_train, treatment_train)

# Прогнозирование uplift на тестовых данных
uplift_preds = uplift_model.predict(X_test_selected)

# Сохранение предсказаний в файл для сдачи
test_data['successful_utilization'] = uplift_preds
filtered_test_data = test_data[['successful_utilization']].copy()
filtered_test_data.to_csv('uplift_predictions.csv', index=True)  # Сохраняем стандартный индекс

# Оценка модели на валидационной выборке
uplift_val_preds = uplift_model.predict(X_val_selected)
score = uplift_at_k(y_val, uplift_val_preds, treatment_val, strategy='overall')

print(f"Uplift score on validation set: {score:.4f}")
