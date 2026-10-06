import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(

    page_title="AI Diabetes Prediction",

    layout="wide"
)

# Custom CSS to force white background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("diabetes_model.pkl")

scaler = joblib.load("scaler.pkl")

feature_names = joblib.load("feature_names.pkl")

# =====================================================
# TITLE
# =====================================================
st.title("Diabetes Prediction System with Neural Network (MLPClassifier)")

st.markdown("""
Sistem prediksi diabetes menggunakan Neural Network (MLPClassifier).

Aplikasi ini menampilkan:
- hasil prediksi
- confidence score
- hasil StandardScaler
- bobot fitur (weights)
- bias neuron
- analisis kontribusi fitur
""")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Input Data Pasien")

input_data = {}

# =====================================================
# INPUT NUMERIK
# =====================================================

input_data["Age"] = st.sidebar.number_input(

    "Age",

    min_value=1,

    max_value=100,

    value=30,

    help="Usia pasien."
)

input_data["BMI"] = st.sidebar.number_input(

    "BMI",

    min_value=10.0,

    max_value=60.0,

    value=24.0,

    help="""
Body Mass Index

Normal:
18.5 - 24.9
"""
)

input_data["FastingBloodSugar"] = st.sidebar.number_input(

    "Fasting Blood Sugar",

    min_value=50.0,

    max_value=300.0,

    value=90.0,

    help="""
Normal:
<100 mg/dL
"""
)

input_data["HbA1c"] = st.sidebar.number_input(

    "HbA1c",

    min_value=3.0,

    max_value=15.0,

    value=5.5,

    help="""
Normal:
<5.7%
"""
)

input_data["CholesterolHDL"] = st.sidebar.number_input(

    "Cholesterol HDL",

    min_value=10.0,

    max_value=150.0,

    value=50.0,

    help="""
HDL merupakan kolesterol baik.
Semakin tinggi nilainya semakin baik.
"""
)

input_data["CholesterolLDL"] = st.sidebar.number_input(

    "Cholesterol LDL",

    min_value=10.0,

    max_value=300.0,

    value=100.0,

    help="""
LDL merupakan kolesterol jahat.
Semakin tinggi nilainya semakin berisiko.
"""
)

# =====================================================
# SLIDER
# =====================================================

input_data["PhysicalActivity"] = st.sidebar.slider(

    "Physical Activity",

    min_value=0,

    max_value=10,

    value=5,

    help="""
0 = Sangat tidak aktif
10 = Sangat aktif
"""
)

input_data["DietQuality"] = st.sidebar.slider(

    "Diet Quality",

    min_value=0,

    max_value=10,

    value=5,

    help="""
0 = Pola makan buruk
10 = Pola makan sangat sehat
"""
)

# =====================================================
# SELECTBOX
# =====================================================

smoking = st.sidebar.selectbox(

    "Smoking",

    ["No", "Yes"],

    help="Status merokok pasien."
)

input_data["Smoking"] = 1 if smoking == "Yes" else 0

family_history = st.sidebar.selectbox(

    "Family History Diabetes",

    ["No", "Yes"],

    help="Riwayat diabetes dalam keluarga."
)

input_data["FamilyHistoryDiabetes"] = (

    1 if family_history == "Yes" else 0
)

hypertension = st.sidebar.selectbox(

    "Hypertension",

    ["No", "Yes"],

    help="Riwayat hipertensi."
)

input_data["Hypertension"] = (

    1 if hypertension == "Yes" else 0
)

# =====================================================
# DATAFRAME
# =====================================================

input_df = pd.DataFrame([input_data])

# Pastikan urutan fitur sama seperti saat training
input_df = input_df[feature_names]

# =====================================================
# TAMPILKAN INPUT
# =====================================================

st.subheader("Data Pasien")

st.dataframe(
    input_df,
    use_container_width=True
)

# =====================================================
# PREDIKSI
# =====================================================

if st.button("Jalankan Prediksi"):

    try:

        # =============================================
        # STANDARD SCALER
        # =============================================
        input_scaled = scaler.transform(input_df)

        st.write("---")

        st.subheader("Hasil StandardScaler")

        scaled_df = pd.DataFrame({

            "Feature": feature_names,

            "Scaled Input": input_scaled[0]

        })

        st.dataframe(

            scaled_df,

            use_container_width=True
        )

        st.caption("""
Nilai di atas merupakan hasil normalisasi menggunakan StandardScaler.

Nilai inilah yang masuk ke dalam jaringan saraf (MLP).
""")

        # =============================================
        # PREDIKSI
        # =============================================
        prediction = model.predict(input_scaled)

        probability = model.predict_proba(input_scaled)

        confidence = np.max(probability) * 100

        st.write("---")

        # =============================================
        # HASIL PREDIKSI
        # =============================================
        st.subheader("Hasil Prediksi")

        if prediction[0] == 1:

            st.error(
                "Risiko Diabetes Tinggi"
            )

        else:

            st.success(
                "Risiko Diabetes Rendah"
            )

        # =============================================
        # CONFIDENCE SCORE
        # =============================================
        st.metric(

            label="Confidence Score",

            value=f"{confidence:.2f}%"
        )

        # =============================================
        # INTERPRETASI CONFIDENCE
        # =============================================
        if confidence >= 90:

            st.success(
                "Model sangat yakin terhadap hasil prediksi."
            )

        elif confidence >= 75:

            st.info(
                "Model cukup yakin terhadap hasil prediksi."
            )

        elif confidence >= 60:

            st.warning(
                "Keyakinan model sedang."
            )

        else:

            st.error(
                "Keyakinan model rendah."
            )

        st.write("---")

                # =================================================
        # WEIGHTS LAYER PERTAMA
        # =================================================
        st.subheader("Bobot Fitur (Weight Importance)")

        # Mengambil bobot dari Input Layer -> Hidden Layer 1
        weights = model.coefs_[0]

        # Menghitung rata-rata absolut bobot setiap fitur
        avg_weights = np.mean(np.abs(weights), axis=1)

        weight_df = pd.DataFrame({

            "Feature": feature_names,

            "Weight Importance": avg_weights

        })

        weight_df = weight_df.sort_values(

            by="Weight Importance",

            ascending=False

        )

        st.dataframe(

            weight_df,

            use_container_width=True
        )

        st.bar_chart(

            weight_df.set_index("Feature")

        )

        st.caption("""
Semakin besar Weight Importance,
semakin besar pengaruh fitur terhadap proses pembelajaran model.
""")

        st.write("---")

        # =================================================
        # BIAS SETIAP LAYER
        # =================================================
        st.subheader("Bias Neuron")

        biases = model.intercepts_

        bias_hidden_1 = biases[0]

        bias_hidden_2 = biases[1]

        bias_output = biases[2]

        st.write("### Hidden Layer 1")

        st.write(bias_hidden_1)

        st.write("### Hidden Layer 2")

        st.write(bias_hidden_2)

        st.write("### Output Layer")

        st.write(bias_output)

        st.caption("""
Bias membantu neuron menyesuaikan keputusan
agar model mampu belajar lebih baik.
""")

        st.write("---")

        # =================================================
        # KONTRIBUSI FITUR
        # =================================================
        st.subheader("Kontribusi Fitur terhadap Prediksi")

        scaled_values = input_scaled[0]

        contribution = scaled_values * avg_weights

        contribution_df = pd.DataFrame({

            "Feature": feature_names,

            "Scaled Input": scaled_values,

            "Weight Importance": avg_weights,

            "Contribution": contribution

        })

        contribution_df = contribution_df.sort_values(

            by="Contribution",

            ascending=False

        )

        st.dataframe(

            contribution_df,

            use_container_width=True
        )

        st.bar_chart(

            contribution_df.set_index("Feature")[

                "Contribution"

            ]
        )

        st.caption("""
Contribution dihitung menggunakan:

Contribution = Scaled Input × Weight Importance

Semakin besar nilai Contribution,
semakin besar pengaruh fitur terhadap prediksi pasien saat ini.
""")
    # =====================================================
    # ERROR HANDLER
    # =====================================================
    except Exception as e:

        st.error(f"Terjadi error: {e}")

# =====================================================
# FOOTER
# =====================================================
st.write("---")

st.caption("""
Developed Using

• Python

• Streamlit

• Scikit-Learn

• Multi Layer Perceptron (MLPClassifier)

• Diabetes Prediction System
""")