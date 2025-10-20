import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from src.FeaturesGenerator import FeaturesGenerator



# read data (Timestamp,Open,High,Low,Close,Volume)
df_original = pd.read_csv("data/btc_2024.csv")
#
df_prepared = df_original.copy()
df_prepared.drop(['Timestamp', 'Volume'], axis=1, inplace=True)


fg = FeaturesGenerator()
# Add target: 3% increase in 45 bars
df_prepared = fg.add_target(df_prepared, 45, 3)
# Add features
df_prepared = fg.add_features(df_prepared)
# Add returns
df_prepared = fg.returnificate(df_prepared)
# Drop rows with NaN
df_prepared.dropna(inplace=True)
# Drop Open, High, Low, Close columns
df_prepared.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

# split data
df_train, df_test = train_test_split(df_prepared, test_size=0.2, random_state=42)

# Define models to train
models = {
    "linear_regression": (
        LinearRegression(),
        {}
    ),
    "ridge_regression": (
        Ridge(alpha=1.0),
        {"alpha": 1.0}
    ),
    "lasso_regression": (
        Lasso(alpha=0.1),
        {"alpha": 0.1}
    ),
    "random_forest": (
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    )
}

# feature scaling
scaler = StandardScaler()
df_train = scaler.fit_transform(df_train)
df_test = scaler.transform(df_test)

#
for model_name, model_data in models.items():
    model = model_data[0]
    params = model_data[1]
    model.fit(df_train, df_train)
    #model.eval()
    y_pred = model.predict(df_test)
    print("Model:", model_name)
    print("Accuracy:", accuracy_score(df_test, y_pred))
    print("F1:", f1_score(df_test, y_pred))
    print("Precision:", precision_score(df_test, y_pred))
    print("Recall:", recall_score(df_test, y_pred))
    print("Cross validation scores:", cross_val_score(model, df_train, df_train, cv=5))
    print("Cross validation accuracy:", cross_val_score(model, df_train, df_train, cv=5).mean())
    print("Cross validation precision:", cross_val_score(model, df_train, df_train, cv=5).mean())
    print("Cross validation recall:", cross_val_score(model, df_train, df_train, cv=5).mean())
    print("Cross validation f1:", cross_val_score(model, df_train, df_train, cv=5).mean())
    print("Cross validation std:", cross_val_score(model, df_train, df_train, cv=5).std())
    print('\n\n')







# train model
model = LogisticRegression()
model.fit(df_train, df_train['Close'])

# evaluate model
y_pred = model.predict(df_test)
print("Accuracy:", accuracy_score(df_test['Close'], y_pred))
print("F1:", f1_score(df_test['Close'], y_pred))
print("Precision:", precision_score(df_test['Close'], y_pred))
print("Recall:", recall_score(df_test['Close'], y_pred))

# cross validation
scores = cross_val_score(model, df_train, df_train['Close'], cv=5)
print("Cross validation scores:", scores)
print("Cross validation accuracy:", scores.mean())

# save model
model.save("models/logistic_regression.joblib")
