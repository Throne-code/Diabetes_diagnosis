import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("diabetes_data.csv")

# =====================================================
# TAMPILKAN KOLOM
# =====================================================
print("\n===== DAFTAR KOLOM =====\n")
print(df.columns.tolist())

# =====================================================
# FITUR PENTING
# =====================================================
selected_features = [

    "Age",
    "BMI",
    "FastingBloodSugar",
    "HbA1c",
    "CholesterolHDL",
    "CholesterolLDL",
    "PhysicalActivity",
    "DietQuality",
    "Smoking",
    "FamilyHistoryDiabetes",
    "Hypertension"
]

# =====================================================
# FILTER FITUR YANG ADA
# =====================================================
available_features = [

    col for col in selected_features

    if col in df.columns
]

print("\n===== FITUR DIGUNAKAN =====\n")
print(available_features)

# =====================================================
# INPUT
# =====================================================
X = df[available_features].copy()

# =====================================================
# TARGET
# =====================================================
if "Diagnosis" not in df.columns:

    raise Exception(
        "Kolom Diagnosis tidak ditemukan."
    )

y = df["Diagnosis"]

# =====================================================
# KONVERSI KE NUMERIK
# =====================================================
for col in X.columns:

    X[col] = pd.to_numeric(

        X[col],

        errors="coerce"
    )

# =====================================================
# CEK MISSING VALUE
# =====================================================
print("\n===== JUMLAH NaN =====\n")
print(X.isnull().sum())

# =====================================================
# HANDLE MISSING VALUE
# =====================================================
X = X.fillna(0)

# =====================================================
# CEK ULANG
# =====================================================
print("\n===== NaN SETELAH FILL =====\n")
print(X.isnull().sum())

# =====================================================
# NORMALISASI DATA
# =====================================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n===== DATA SETELAH STANDARD SCALER =====\n")

scaled_df = pd.DataFrame(

    X_scaled,

    columns=available_features
)

print(scaled_df.head())

# =====================================================
# TAMPILKAN MEAN DAN STANDAR DEVIASI
# =====================================================
print("\n===== PARAMETER STANDARD SCALER =====\n")

scaler_info = pd.DataFrame({

    "Feature": available_features,

    "Mean": scaler.mean_,

    "Std": scaler.scale_
})

print(scaler_info)

# =====================================================
# SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.2,

    random_state=42
)

print("\n===== JUMLAH DATA =====\n")

print("Training :", len(X_train))

print("Testing  :", len(X_test))

# =====================================================
# MODEL MLP
# =====================================================
model = MLPClassifier(

    hidden_layer_sizes=(64, 32),

    activation="relu",

    learning_rate_init=0.001,

    max_iter=2000,

    random_state=42
)

# =====================================================
# TRAINING
# =====================================================
print("\n===== TRAINING MODEL =====\n")

model.fit(X_train, y_train)

print("Training selesai.")

# =====================================================
# PREDIKSI
# =====================================================
predictions = model.predict(X_test)

accuracy = accuracy_score(

    y_test,

    predictions
)

print(f"\nAkurasi Model : {accuracy * 100:.2f}%")

# =====================================================
# INFORMASI MODEL
# =====================================================
print("\n===== INFORMASI MODEL =====\n")

print("Jumlah Hidden Layer :", len(model.hidden_layer_sizes))

print("Neuron Hidden Layer :", model.hidden_layer_sizes)

print("Fungsi Aktivasi     :", model.activation)

print("Learning Rate       :", model.learning_rate_init)

print("Iterasi Maksimum    :", model.max_iter)

print("Iterasi Aktual      :", model.n_iter_)

# =====================================================
# INFORMASI WEIGHT
# =====================================================
print("\n===== SHAPE WEIGHT =====\n")

for i, weight in enumerate(model.coefs_):

    print(f"Layer {i+1} : {weight.shape}")

# =====================================================
# INFORMASI BIAS
# =====================================================
print("\n===== SHAPE BIAS =====\n")

for i, bias in enumerate(model.intercepts_):

    print(f"Layer {i+1} : {bias.shape}")

# =====================================================
# SIMPAN MODEL
# =====================================================
joblib.dump(

    model,

    "diabetes_model.pkl"
)

joblib.dump(

    scaler,

    "scaler.pkl"
)

joblib.dump(

    list(X.columns),

    "feature_names.pkl"
)

print("\n===== MODEL BERHASIL DISIMPAN =====\n")

print("- diabetes_model.pkl")

print("- scaler.pkl")

print("- feature_names.pkl")