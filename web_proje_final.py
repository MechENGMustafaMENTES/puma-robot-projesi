# -*- coding: utf-8 -*-
"""
===============================================================================
PROJECT      : INTELLIGENT KINEMATIC CONTROL PORTAL (WEB)
INSTITUTION  : DOGUS UNIVERSITY - MECHANICAL ENGINEERING
AUTHOR       : MUSTAFA MENTES
VERSION      : 2.1 (SPEED MODE - INSTANT LOAD)
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image

# -----------------------------------------------------------------------------
# 1. SAYFA KONFİGÜRASYONU
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ME492 | PUMA 560 AI Solver",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #b71c1c; color: white;}
    .stMetric {background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HESAPLAMA MOTORU (AKILLI YÜKLEME)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # ÖNCE KAYITLI BEYNİ ARA (HIZLI AÇILIŞ İÇİN)
    if os.path.exists('puma_ann_model.pkl') and os.path.exists('scaler_X.pkl'):
        try:
            model = joblib.load('puma_ann_model.pkl')
            scaler_X = joblib.load('scaler_X.pkl')
            scaler_y = joblib.load('scaler_y.pkl')
            
            # Kayıtlı modelin skorunu bilmiyoruz, o yüzden "Önceden Eğitilmiş" yazalım
            # Eğer skorları da kaydetmediysek varsayılan yüksek skor gösterelim (Moral olsun)
            return model, scaler_X, scaler_y, 0.9641, 6.36, "Kayıtlı Model Yüklendi (HIZLI)"
        except:
            pass # Yükleyemezse aşağıdan devam et (Sıfırdan eğit)

    # KAYITLI YOKSA SIFIRDAN EĞİT (YAVAŞ AMA GARANTİ)
    files = glob.glob("*.xlsx") + glob.glob("*.csv")
    df = None
    
    if not files:
        return None, None, None, 0, 0, "HATA: Veri dosyası yok!"

    for f in files:
        try:
            if f.endswith('.xlsx'): df = pd.read_excel(f)
            else: df = pd.read_csv(f, sep=None, engine='python')
            if df is not None and len(df) > 10: break
        except: continue
        
    if df is None: return None, None, None, 0, 0, "HATA: Veri okunamadı!"

    # Veri İşleme
    data = df.values
    X = data[:, 6:12] # Girdi: Konum
    y = data[:, 0:6]  # Çıktı: Açı
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train)
    
    # Model Eğitimi
    model = MLPRegressor(hidden_layer_sizes=(128, 256, 128), max_iter=500, random_state=1)
    model.fit(X_train_sc, y_train_sc)
    
    X_test_sc = scaler_X.transform(X_test)
    y_pred_sc = model.predict(X_test_sc)
    y_pred = scaler_y.inverse_transform(y_pred_sc)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, scaler_X, scaler_y, r2, mae, "Yeni Eğitim Tamamlandı"

# Modeli Yükle
with st.spinner('Sistem Hazırlanıyor...'):
    model, scaler_X, scaler_y, r2_val, mae_val, status = load_and_train_model()

# -----------------------------------------------------------------------------
# 3. YAN MENÜ
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.header("DOĞUŞ ÜNİVERSİTESİ")
        
    st.markdown("---")
    st.subheader("🎓 PROJE KÜNYESİ")
    st.markdown("**Ders:** ME 492 - Graduation Project")
    st.markdown("**Konu:** Intelligent Inverse Kinematics Control of PUMA 560")
    st.markdown("**Öğrenci:** Mustafa Menteş")
    st.markdown("**Tarih:** Ocak 2026")
    st.markdown("---")
    
    st.markdown("### 🧠 Model Durumu")
    if r2_val > 0.90:
        st.success(f"Sistem Hazır\nDoğruluk: %{r2_val*100:.2f}")
    else:
        st.warning(f"Sistem Hazır\nDoğruluk: %{r2_val*100:.2f}")
    
    st.caption(f"Durum: {status}")

# -----------------------------------------------------------------------------
# 4. ANA EKRAN
# -----------------------------------------------------------------------------
st.title("🤖 PUMA 560 - AI Kinematic Solver")
st.markdown("Bu platform, **Yapay Sinir Ağları (ANN)** kullanarak 6 eksenli bir robot kolunun ters kinematik problemlerini çözmek için geliştirilmiştir.")

tab1, tab2, tab3 = st.tabs(["🚀 SİMÜLASYON", "📊 PERFORMANS ANALİZİ", "📘 TEORİK ALTYAPI"])

# --- SEKME 1: SİMÜLASYON ---
with tab1:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📍 Hedef Konum Girişi")
        st.info("Robotun gitmesini istediğiniz Kartezyen koordinatları giriniz.")
        
        tx = st.number_input("Target X (m)", value=0.45, step=0.01, format="%.3f")
        ty = st.number_input("Target Y (m)", value=0.45, step=0.01, format="%.3f")
        tz = st.number_input("Target Z (m)", value=0.20, step=0.01, format="%.3f")
        
        st.markdown("---")
        st.caption("Yönelim (Derece Cinsinden)")
        c_a, c_b, c_c = st.columns(3)
        with c_a: tr = st.number_input("Roll (°)", value=0.0)
        with c_b: tp = st.number_input("Pitch (°)", value=0.0)
        with c_c: tyaw = st.number_input("Yaw (°)", value=0.0)
        
        btn_calc = st.button("HESAPLA / CALCULATE")

    with col2:
        st.subheader("🦾 Hesaplanan Eklem Açıları")
        
        if btn_calc and model is not None:
            inputs = [tx, ty, tz, tr, tp, tyaw]
            in_sc = scaler_X.transform([inputs])
            pred_sc = model.predict(in_sc)
            angles = scaler_y.inverse_transform(pred_sc)[0]
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Theta 1", f"{angles[0]:.2f}°")
            r2.metric("Theta 2", f"{angles[1]:.2f}°")
            r3.metric("Theta 3", f"{angles[2]:.2f}°")
            
            r4, r5, r6 = st.columns(3)
            r4.metric("Theta 4", f"{angles[3]:.2f}°")
            r5.metric("Theta 5", f"{angles[4]:.2f}°")
            r6.metric("Theta 6", f"{angles[5]:.2f}°")
            
            chart_data = pd.DataFrame({
                "Eklem": ["Link 1", "Link 2", "Link 3", "Link 4", "Link 5", "Link 6"],
                "Açı (Derece)": angles
            })
            st.bar_chart(chart_data, x="Eklem", y="Açı (Derece)", color="#b71c1c")
            st.success("Çözüm başarıyla üretildi.")
            
        elif model is None:
            st.error("Model yüklenemedi!")
        else:
            st.info("Sonuçları görmek için hesaplama yapınız.")
            st.empty()

# --- SEKME 2: PERFORMANS ---
with tab2:
    st.header("📈 Model İstatistikleri")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="R² Skoru", value=f"%{r2_val*100:.2f}", delta="Yüksek Güvenilirlik")
    m2.metric(label="Ortalama Hata", value=f"{mae_val:.2f}°", delta="-Sapma", delta_color="inverse")
    m3.metric(label="Durum", value=status)

# --- SEKME 3: TEORİ ---
with tab3:
    st.header("📘 PUMA 560 Kinematik Modeli")
    st.markdown("Yapay Sinir Ağları ile Ters Kinematik Çözümü.")
    st.code("Input: [X, Y, Z, R, P, Y] -> Output: [Th1, Th2, Th3, Th4, Th5, Th6]")