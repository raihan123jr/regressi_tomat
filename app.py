import streamlit as st
import pandas as pd
import joblib

st.sidebar.title("Machine Learning")
st.sidebar.success("Dibuat Oleh MARMAR")

st.title("Regressi Penjualan Tomat")
st.markdown("Aplikasi machine learning regression untuk menghitung tital penjualan tomat berdasarkan fitur `harga,hari,cuaca,dan promo`")

model_forest = joblib.load("model_forest.joblib")

Harga = st.slider("Harga",0,20000,10000)
Hari = st.selectbox("Hari",["Senin","Selasa","Rabu","Kamis","Jum'at","Sabtu","Minggu"])
Cuaca = st.selectbox("Cuaca",["Cerah","Bearawan","Mendung","Hujan"])
Promo = st.pills("Promo",["Ya","Tidak"],default="Tidak")

if st.button("prediksi"):
    data_baru = pd.DataFrame([[Harga,Hari,Cuaca,Promo]],
columns=["Harga","Hari","Cuaca","Promo"])
    prediksi = model_forest.predict(data_baru)[0]
    st.success(f"Model Memprediksi total penjualannya **{prediksi:.0f}**")
    st.balloons()