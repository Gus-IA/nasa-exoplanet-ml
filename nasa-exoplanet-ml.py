import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# URL del dataset KOI (Kepler Objects of Interest)
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"

# Descargar dataset
df = pd.read_csv(url)

# Mostrar primeras filas
print(df.head())

# Guardar localmente
df.to_csv(df.to_csv("kepler_koi_dataset.csv", index=False))

print(f"\nDataset descargado correctamente.")
print(f"Shape Filas y columnas: {df.shape}")
print(f"Columnas: {df.columns}")
print(df["koi_disposition"].value_counts())
print(df.isnull().sum().sort_values(ascending=False).head(20))

target = "koi_disposition"

df = df[df[target] != "CANDIDATE"]  # solo binario
y = df[target]

# quitamos columnas
drop_cols = ["rowid", "kepid", "kepoi_name"]

df = df.drop(columns=drop_cols, errors="ignore")

# quitamos el target de X
X = df.drop(columns=["koi_disposition"])

# valores nulos
imputer = SimpleImputer(strategy="median")

X = X.select_dtypes(include=[np.number])  # solo numéricas

X_imputed = imputer.fit_transform(X)

# codificamos target
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# split

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# train
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)


# predictions
y_pred = model.predict(X_test)

# eval
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:\n")
print(confusion_matrix(y_test, y_pred))


# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

xgb.fit(X_train, y_train)

# predictions
y_pred_xgb = xgb.predict(X_test)

# eval
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# grid search

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 300, 500],
    "subsample": [0.8, 1.0],
}

grid = GridSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
)

grid.fit(X_train, y_train)


best_model = grid.best_estimator_

print(grid.best_params_)
