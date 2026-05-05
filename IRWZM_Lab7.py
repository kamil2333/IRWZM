import os
import numpy as np
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

DATA_PATH = "health_measurements.csv"
MODEL_PATH = "risk_model.joblib"

st.set_page_config(page_title="Monitor zdrowia + ML", layout="centered")
st.title("📱 Monitor zdrowia + analiza ML (Wariant 4)")

# -----------------------------
# Pomocnicze: inicjalizacja CSV
# -----------------------------
def ensure_data_file():
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame(columns=[
            "timestamp", "age", "bmi", "glucose", "systolic_bp", "diastolic_bp", "pulse", "steps"
        ])
        df.to_csv(DATA_PATH, index=False)
    else:
        # Aktualizacja istniejącego pliku z poprzednich zajęć o nowe kolumny
        df = pd.read_csv(DATA_PATH)
        added = False
        if "pulse" not in df.columns:
            df["pulse"] = 70
            added = True
        if "steps" not in df.columns:
            df["steps"] = 5000
            added = True
        if added:
            df.to_csv(DATA_PATH, index=False)

def load_data():
    ensure_data_file()
    return pd.read_csv(DATA_PATH)

def append_measurement(row: dict):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

def make_demo_label(df: pd.DataFrame) -> pd.Series:
    """
    Etykieta wieloklasowa (3 poziomy ryzyka):
    2 (Wysokie)     -> SBP >= 140 lub DBP >= 90
    1 (Umiarkowane) -> SBP >= 120 lub DBP >= 80 (i mniejsze od progów wyżej)
    0 (Niskie)      -> pozostałe (SBP < 120 i DBP < 80)
    """
    conditions = [
        (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90),
        (df["systolic_bp"] >= 120) | (df["diastolic_bp"] >= 80)
    ]
    choices = [2, 1]
    return pd.Series(np.select(conditions, choices, default=0))

def train_model(df: pd.DataFrame):
    if len(df) < 20:
        raise ValueError("Za mało danych do trenowania (min. 20 pomiarów). Dodaj więcej wpisów.")

    y = make_demo_label(df)
    X = df[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp", "pulse", "steps"]].copy()

    # Usunięto stratify=y, ponieważ w małych, sztucznych zbiorach
    # przy 3 klasach jedna klasa może wystąpić tylko raz i zepsuć podział.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    num_cols = list(X.columns)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    # Regresja logistyczna dla wielu klas (scikit-learn teraz robi to automatycznie)
    clf = Pipeline(steps=[
        ("pre", pre),
        ("model", LogisticRegression(solver='lbfgs', max_iter=2000))
    ])

    clf.fit(X_train, y_train)

    # Metryki
    proba = clf.predict_proba(X_test)
    pred = clf.predict(X_test)

    # Obliczanie ROC AUC dla klasyfikacji wieloklasowej
    try:
        auc = float(roc_auc_score(y_test, proba, multi_class="ovr"))
    except Exception:
        auc = None # w razie braku wszystkich klas w zbiorze testowym

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, digits=3, zero_division=0)
    }

    joblib.dump({"model": clf, "metrics": metrics}, MODEL_PATH)
    return clf, metrics

def load_model():
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj["model"], obj["metrics"]
    return None, None

# =========================
# ETAP 1: Zbieranie danych
# =========================
st.header("Etap 1 — Zbieranie danych zdrowotnych")

with st.form("health_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Wiek [lata]", min_value=18, max_value=110, value=40, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        glucose = st.number_input("Glukoza [mg/dl]", min_value=40, max_value=300, value=95, step=1)
        pulse = st.number_input("Tętno [ud/min]", min_value=40, max_value=200, value=70, step=1)
    with col2:
        systolic_bp = st.number_input("Ciśnienie skurczowe SBP [mmHg]", min_value=70, max_value=260, value=120, step=1)
        diastolic_bp = st.number_input("Ciśnienie rozkurczowe DBP [mmHg]", min_value=40, max_value=150, value=80, step=1)
        steps = st.number_input("Kroki dzienne", min_value=0, max_value=100000, value=5000, step=100)

    submitted = st.form_submit_button("💾 Zapisz pomiar")

if submitted:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "age": int(age),
        "bmi": float(bmi),
        "glucose": int(glucose),
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
        "pulse": int(pulse),
        "steps": int(steps)
    }
    append_measurement(row)
    st.success("Zapisano pomiar do pliku health_measurements.csv")

df = load_data()
st.caption(f"Liczba zapisanych pomiarów: {len(df)}")
st.dataframe(df.tail(10), use_container_width=True)

# =====================================
# ETAP 2: Analiza i wizualizacja danych
# =====================================
st.header("Etap 2 — Analiza i wizualizacja")

if len(df) == 0:
    st.info("Dodaj co najmniej jeden pomiar, aby zobaczyć analizę.")
else:
    st.subheader("Wykres trendu (ostatnie pomiary)")
    plot_cols = st.multiselect(
        "Wybierz parametry do wykresu:",
        options=["bmi", "glucose", "systolic_bp", "diastolic_bp", "pulse", "steps"],
        default=["systolic_bp", "diastolic_bp"]
    )

    if plot_cols:
        df_plot = df.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
        df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp").tail(50)

        fig = plt.figure(figsize=(7, 4))
        for c in plot_cols:
            plt.plot(df_plot["timestamp"], df_plot[c], label=c)
        plt.xlabel("czas")
        plt.ylabel("wartość")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # NOWE: Wizualizacja rozkładu klas ryzyka
    st.subheader("Rozkład klas ryzyka (bazujący na progach)")
    df_risk = df.copy()
    df_risk["Risk_Level"] = make_demo_label(df_risk).map({0: "0 (Niskie)", 1: "1 (Umiarkowane)", 2: "2 (Wysokie)"})
    risk_counts = df_risk["Risk_Level"].value_counts().sort_index()
    st.bar_chart(risk_counts)

# ==============================
# ETAP 3: Model uczenia maszynowego
# ==============================
st.header("Etap 3 — Budowa modelu ML (Wieloklasowy)")

st.write(
    "Klasyfikator posiada 3 klasy: Niskie (0), Umiarkowane (1) i Wysokie (2) ryzyko. "
    "Trenowany jest model `LogisticRegression(multi_class='multinomial')`."
)

model, metrics = load_model()

colA, colB = st.columns([1, 2])
with colA:
    if st.button("🧠 Wytrenuj / odśwież model"):
        try:
            model, metrics = train_model(df)
            st.success("Model został wytrenowany i zapisany.")
        except Exception as e:
            st.error(str(e))

with colB:
    if metrics:
        st.subheader("Metryki (na części testowej)")
        st.write(f"Accuracy: **{metrics['accuracy']:.3f}**")
        if metrics["roc_auc"] is not None:
            st.write(f"ROC AUC (OVR): **{metrics['roc_auc']:.3f}**")
        st.text("Classification report:\n" + metrics["report"])
    else:
        st.info("Model nie jest jeszcze wytrenowany. Kliknij przycisk obok.")

# ===================================
# ETAP 4: Integracja modelu z aplikacją
# ===================================
st.header("Etap 4 — Predykcja w aplikacji (Klasy Ryzyka)")

if model is None:
    st.warning("Najpierw wytrenuj model w Etapie 3.")
else:
    st.subheader("Predykcja ryzyka dla bieżącego pomiaru")
    X_one = pd.DataFrame([{
        "age": int(age),
        "bmi": float(bmi),
        "glucose": int(glucose),
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
        "pulse": int(pulse),
        "steps": int(steps)
    }])

    # Model zwraca tablicę z prawdopodobieństwami dla każdej z 3 klas
    proba = model.predict_proba(X_one)[0]
    pred = int(model.predict(X_one)[0])

    st.write("Prawdopodobieństwa przypisane przez model ML:")
    st.write(f"🟢 Niskie: **{proba[0]:.2%}** | 🟡 Umiarkowane: **{proba[1]:.2%}** | 🔴 Wysokie: **{proba[2]:.2%}**")

    # Interpretacja klas i krótkie zalecenia edukacyjne
    if pred == 0:
        st.success("🩺 **Klasa 0: Niskie ryzyko**\n\nTwoje parametry wydają się być w normie. Edukacyjnie: Utrzymuj zdrową dietę i regularnie uprawiaj sport, aby utrzymać ten stan!")
    elif pred == 1:
        st.warning("🩺 **Klasa 1: Umiarkowane ryzyko**\n\nZauważono lekkie odchylenia. Edukacyjnie: Zwróć szczególną uwagę na spożycie soli, zadbaj o regularną aktywność fizyczną i częściej monitoruj ciśnienie.")
    else:
        st.error("🩺 **Klasa 2: Wysokie ryzyko**\n\nZnaczne przekroczenia norm! Edukacyjnie: Skonsultuj swoje wyniki z lekarzem i zrób badania kontrolne.")

    st.caption("Uwaga: to demonstracja edukacyjna. Nie jest to wyrób medyczny ani narzędzie diagnostyczne.")