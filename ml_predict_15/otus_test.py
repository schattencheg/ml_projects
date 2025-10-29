import pandas as pd
from src.FeaturesGenerator import FeaturesGenerator
# Data paths
PATH_TRAIN = "data/hour/btc.csv"
PATH_TRAIN = "data/sber_data.csv"
PATH_TRAIN = "data/minute/btc_2025.csv"


print("Loading training data...")
df_train = pd.read_csv(PATH_TRAIN)
fg = FeaturesGenerator()
X_train, y_train, X_val, y_val, X_test, y_test, new_trend_features = fg.add_features_otus(df_train)
features = new_trend_features

def calculate_metrics_table(y_true, y_pred_prob, thresholds=[0.5, 0.6, 0.7, 0.8]):
    metrics_table = []
    
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        metrics = {
            'Cutoff': threshold,
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred)
        }
        metrics_table.append(metrics)
    
    # Преобразуем список словарей в DataFrame для удобного вывода
    return pd.DataFrame(metrics_table)*100

# Выводим ROC AUC и метрики для всех выборок
def display_metrics_set(name, y_true, y_pred_prob):
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\n=== Метрики для {name} выборки ===")
    print(f"ROC AUC: {roc_auc:.4f}")
    metrics_table = calculate_metrics_table(y_true, y_pred_prob)
    print(metrics_table)



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score

# 1. Применение MinMaxScaler к тренировочной, валидационной и тестовой выборкам
scaler = MinMaxScaler()

# Обучаем скейлер на тренировочной выборке и трансформируем её
X_train_scaled = scaler.fit_transform(X_train)

# Трансформируем валидационную и тестовую выборки
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 2. Обучение модели логистической регрессии на тренировочной выборке
model = LogisticRegression(solver='liblinear')  # Используем liblinear для бинарной классификации
model.fit(X_train_scaled, y_train)

# 3. Предсказания на валидационной и тестовой выборках
y_train_pred_prob = model.predict_proba(X_train_scaled)[:, 1]
y_val_pred_prob = model.predict_proba(X_val_scaled)[:, 1]
y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# 4. Выводим ROC AUC для валидационной и тестовой выборок
train_roc_auc = roc_auc_score(y_train, y_train_pred_prob)
val_roc_auc = roc_auc_score(y_val, y_val_pred_prob)
test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)

print("\n=== Метрики для валидационной выборки ===")
print(f"ROC AUC: {train_roc_auc:.4f}")
print(calculate_metrics_table(y_train, y_train_pred_prob))

print("\n=== Метрики для валидационной выборки ===")
print(f"ROC AUC: {val_roc_auc:.4f}")
print(calculate_metrics_table(y_val, y_val_pred_prob))

print("\n=== Метрики для тестовой выборки ===")
print(f"ROC AUC: {test_roc_auc:.4f}")
print(calculate_metrics_table(y_test, y_test_pred_prob))

# 5. Выводим топ-5 самых значимых факторов по весам
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_[0]
})

# Сортируем по абсолютной величине коэффициентов и выбираем топ-5
top_5_features = feature_importances.reindex(feature_importances['Importance'].abs().sort_values(ascending=False).index).head(5)
print("\n=== Топ-5 самых значимых факторов ===")
print(top_5_features)

