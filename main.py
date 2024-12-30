import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('data.csv')
# Load model and scaler
with open("mlp.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Fungsi untuk input data pengguna
def air():
    st.sidebar.header('Input Data')
    st.write("<h2 style='color:red;'>Silahkan Input data</h2>", unsafe_allow_html=True)

    # Input fitur dari pengguna
    Temperature = st.number_input('Temperature', min_value=0.0, max_value=1000.0, value=13.4)
    Humidity = st.number_input('Humidity', min_value=0.0, max_value=1000.0, value=59.1)
    PM2_5 = st.number_input('PM2.5', min_value=0.0, max_value=1000.0, value=5.2)
    PM10 = st.number_input('PM10', min_value=0.0, max_value=1000.0, value=13.4)
    NO2 = st.number_input('NO2', min_value=0.0, max_value=1000.0, value=18.9)
    SO2 = st.number_input('SO2', min_value=0.0, max_value=1000.0, value=9.2)
    CO = st.number_input('CO', min_value=0.0, max_value=1000.0, value=1.72)
    Proximity_to_Industrial_Areas = st.number_input('Proximity to Industrial Areas', min_value=0.0, max_value=1000.0, value=6.3)
    Population_Density = st.number_input('Population Density', min_value=0.0, max_value=1000.0, value=319.0)

    # Membuat DataFrame untuk input fitur
    data = {
        'Temperature': [Temperature],
        'Humidity': [Humidity],
        'PM2.5': [PM2_5],
        'PM10': [PM10],
        'NO2': [NO2],
        'SO2': [SO2],
        'CO': [CO],
        'Proximity_to_Industrial_Areas': [Proximity_to_Industrial_Areas],
        'Population_Density': [Population_Density]
    }
    features = pd.DataFrame(data)
    return features

# Fungsi untuk menampilkan informasi tentang data
def data():
    st.sidebar.header('About')
    st.image('p1.jpg', width=600)
    st.write("<h2 style='color:red;'>Latar Belakang</h2>", unsafe_allow_html=True)


    st.write("""
        Polusi udara menjadi salah satu isu utama yang memengaruhi kualitas hidup 
        manusia di berbagai kota besar di dunia. Tingkat polusi yang tinggi dapat 
        menyebabkan masalah kesehatan serius seperti gangguan pernapasan, penyakit 
        kardiovaskular, dan penurunan kualitas lingkungan secara keseluruhan. 
        Oleh karena itu, pemantauan dan prediksi kualitas udara menjadi hal 
        yang sangat penting untuk memberikan peringatan dini dan membantu 
        pengambilan keputusan kebijakan lingkungan.

        Teknologi sensor gas telah digunakan secara luas untuk memantau kualitas udara. 
        Sensor ini memberikan data respons kimia dari berbagai komponen polutan. 
        Namun, untuk menghasilkan estimasi yang akurat, diperlukan proses kalibrasi
        yang baik dan analisis data lanjutan menggunakan metode pembelajaran mesin.
    """)

    st.write("<h2 style='color:red;'>Data Understanding</h2>", unsafe_allow_html=True)
    st.write("DataFrame")
    st.dataframe(df.head())
    st.write("Data Summary")
    st.dataframe(df.describe())
    
    

# Sidebar untuk memilih opsi
option = st.sidebar.selectbox("Pilih Menu", ["Air Quality", "About Data"])

# Menjalankan fungsi sesuai pilihan
if option == "Air Quality":
    df = air()

    st.subheader('Fitur Input Pengguna')
    st.write(df)

    # Tombol prediksi
    if st.button('Prediksi'):
        try:
            # Pastikan fitur sesuai dengan model
            expected_features = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
            df = df[expected_features]

            # Standardisasi data
            df_scaled = scaler.transform(df)

            # Prediksi probabilitas
            prediction_proba = model.predict_proba(df_scaled)

            # Mapping hasil prediksi
            index_levels = {0: 'Poor', 1: 'Hazardous', 2: 'Good', 3: 'Moderate'}
            index_level = index_levels[prediction_proba.argmax()]

            # Menampilkan hasil prediksi
            st.write(f'Kualitas Udara: {index_level}')
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if option == "About Data":
    data()
