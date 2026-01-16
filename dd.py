import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os

import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from details import st_style, footer, head, about_diabets, warn, mrk

st.set_page_config(
    page_title="Diabetes & Insulin Prediction",
    page_icon="icon.png",
    layout="wide"
)

st.markdown(st_style, unsafe_allow_html=True)
st.markdown(head, unsafe_allow_html=True)

# === Chargement des données de classification ===
df_classif = pd.read_csv("diabetes.csv", header=None)
X_cls = df_classif.iloc[:, :-1].values
y_cls = df_classif.iloc[:, -1].values

# === Chargement et prétraitement des données de régression ===
df_reg = pd.read_csv("diabetes_avec_bolus_filtre333.csv", sep=';')
df_reg.columns = df_reg.columns.str.strip().str.lower()
df_reg = df_reg[df_reg['outcome'] == 1].copy()
df_reg = df_reg.dropna()

# Utilisation des mêmes features que l'entrée utilisateur
features = ["pregnancies", "glucose", "bloodpressure", "skinthickness", "insulin", "bmi", "diabetespedigreefunction", "age"]
X_reg = df_reg[features]
y_reg = df_reg[['bolus_estime', 'basal_dose']]

# === Normalisation et division
scaler_cls = StandardScaler()
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
X_cls_train = scaler_cls.fit_transform(X_cls_train)
X_cls_test = scaler_cls.transform(X_cls_test)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler_reg = StandardScaler()
X_reg_train = scaler_reg.fit_transform(X_reg_train)
X_reg_test = scaler_reg.transform(X_reg_test)

# === Modèle ANN pour la classification
@st.cache_resource
def train_ann_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_cls_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_cls_train, y_cls_train, epochs=80, validation_split=0.2, verbose=0)
    return model

ann_model = train_ann_model()

# === Random Forest pour la prédiction des doses
@st.cache_resource
def train_rf_model():
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_reg_train, y_reg_train)
    return model

rf_model = train_rf_model()

# === Interface utilisateur ===
st.sidebar.title("📝 Enter Clinical Data")
user_input = []

for feature in features:
    label = f"Enter value for `{feature.capitalize()}`"
    if feature in ["bmi", "diabetespedigreefunction"]:
        val = st.sidebar.number_input(label, min_value=0.0, format="%.2f", value=0.0)
    else:
        val = st.sidebar.number_input(label, min_value=0, value=0)
    user_input.append(val)

if any(user_input):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled_cls = scaler_cls.transform(input_array)
        pred_diabetes = ann_model.predict(input_scaled_cls)[0][0]

        fig = go.Figure(data=[go.Pie(
            labels=['Non Diabetic', 'Diabetic'],
            values=[1 - pred_diabetes, pred_diabetes],
            hole=0.6,
            marker=dict(colors=['#1ABC9C', '#E74C3C'])
        )])
        fig.update_layout(title_text="Diabetes Probability", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        if pred_diabetes > 0.5:
            st.markdown(mrk.format("#27AE60", "✅ The person is likely diabetic."), unsafe_allow_html=True)
            input_scaled_reg = scaler_reg.transform(input_array)
            pred_bolus, pred_basal = rf_model.predict(input_scaled_reg)[0]
            st.subheader(f"💉 Estimated Bolus Dose: `{pred_bolus:.2f}` units")
            st.subheader(f"🩺 Estimated Basal Dose: `{pred_basal:.2f}` units")
            
            if st.button("Calculate the Ratio "):
              os.system("streamlit run Ratio.py")
        else:
            st.markdown(mrk.format("#E74C3C", "❌ The person is likely non-diabetic."), unsafe_allow_html=True)
            st.info("No bolus dose required.")

    except Exception as e:
        st.error(f"Error: {e}")

# === Évaluation du modèle ANN
y_pred_prob = ann_model.predict(X_cls_test)
y_pred_cls = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_cls_test, y_pred_cls)
prec = precision_score(y_cls_test, y_pred_cls)
rec = recall_score(y_cls_test, y_pred_cls)
f1 = f1_score(y_cls_test, y_pred_cls)

st.subheader("📊 ANN Classification Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc*100:.2f}%")
col2.metric("Precision", f"{prec*100:.2f}%")
col3.metric("Recall", f"{rec*100:.2f}%")
col4.metric("F1 Score", f"{f1*100:.2f}%")

def show_donut(label, value):
    fig = go.Figure(data=[go.Pie(
        labels=[label, 'Rest'],
        values=[value, 1 - value],
        hole=.7,
        marker_colors=["#00BFFF", "#E0E0E0"],
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        annotations=[dict(text=f"{value*100:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=250, width=250, margin=dict(t=10, b=10, l=10, r=10)
    )
    return fig

st.subheader("🔘 Metrics Visualization (Donuts)")
d1, d2, d3, d4 = st.columns(4)
d1.plotly_chart(show_donut("Accuracy", acc), use_container_width=True)
d2.plotly_chart(show_donut("Precision", prec), use_container_width=True)
d3.plotly_chart(show_donut("Recall", rec), use_container_width=True)
d4.plotly_chart(show_donut("F1 Score", f1), use_container_width=True)
