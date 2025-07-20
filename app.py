import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Judul Dashboard
st.title("🌦️ Prediksi Cuaca di Wilayah Indonesia dengan Machine Learning")
st.write("Upload data cuaca harian untuk melatih model dan prediksi cuaca 10–50 tahun ke depan.")

# Upload File
uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Baca dan proses data
    df = pd.read_excel(uploaded_file, sheet_name='Data Harian - Table')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    # Agregasi bulanan
    cuaca_df = df[['Tahun', 'Bulan', 'Tavg', 'curah_hujan']]
    monthly_df = cuaca_df.groupby(['Tahun', 'Bulan']).agg({
        'Tavg': 'mean',
        'curah_hujan': 'sum'
    }).reset_index()

    st.subheader("📊 Data Bulanan")
    st.dataframe(monthly_df)

    # Split data
    X = monthly_df[['Tahun', 'Bulan']]
    y_temp = monthly_df['Tavg']
    y_rain = monthly_df['curah_hujan']

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rain, test_size=0.2, random_state=42)

    # Model training
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rain = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X_train_t, y_train_t)
    model_rain.fit(X_train_r, y_train_r)

    # Evaluasi model
    pred_temp = model_temp.predict(X_test_t)
    pred_rain = model_rain.predict(X_test_r)

    rmse_temp = np.sqrt(mean_squared_error(y_test_t, pred_temp))
    rmse_rain = np.sqrt(mean_squared_error(y_test_r, pred_rain))
    r2_temp = r2_score(y_test_t, pred_temp)
    r2_rain = r2_score(y_test_r, pred_rain)

    st.subheader("📈 Evaluasi Model")
    st.write(f"**RMSE Suhu**: {rmse_temp:.2f} | **R² Suhu**: {r2_temp:.2f}")
    st.write(f"**RMSE Hujan**: {rmse_rain:.2f} | **R² Hujan**: {r2_rain:.2f}")

    # Prediksi manual (1 bulan)
    st.subheader("🔮 Prediksi Cuaca")
    tahun_input = st.number_input("Masukkan Tahun Prediksi (contoh: 2035)", min_value=2025, max_value=2100, value=2035)
    bulan_input = st.selectbox("Pilih Bulan", list(range(1, 13)))
    input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])
    pred_temp_future = model_temp.predict(input_data)[0]
    pred_rain_future = model_rain.predict(input_data)[0]
    st.success(f"🌡️ Prediksi Suhu Rata-rata {bulan_input}/{tahun_input}: **{pred_temp_future:.2f}°C**")
    st.success(f"🌧️ Prediksi Curah Hujan {bulan_input}/{tahun_input}: **{pred_rain_future:.2f} mm**")

    # Prediksi otomatis 2025–2075
    st.subheader("📆 Prediksi Otomatis 2025–2075")
    future_years = list(range(2025, 2076))
    future_months = list(range(1, 13))
    future_data = pd.DataFrame(
        [(year, month) for year in future_years for month in future_months],
        columns=['Tahun', 'Bulan']
    )
    future_data['Pred_Tavg'] = model_temp.predict(future_data[['Tahun', 'Bulan']])
    future_data['Pred_CurahHujan'] = model_rain.predict(future_data[['Tahun', 'Bulan']])

    st.dataframe(future_data.head(12))

    # Gabung data historis dan prediksi untuk grafik
    monthly_df['Sumber'] = 'Data Historis'
    future_data['Sumber'] = 'Prediksi'
    future_data_merged = pd.concat([
        monthly_df[['Tahun', 'Bulan', 'Tavg', 'curah_hujan', 'Sumber']].rename(columns={'Tavg': 'Suhu', 'curah_hujan': 'CurahHujan'}),
        future_data[['Tahun', 'Bulan', 'Pred_Tavg', 'Pred_CurahHujan', 'Sumber']].rename(columns={'Pred_Tavg': 'Suhu', 'Pred_CurahHujan': 'CurahHujan'})
    ])

    future_data_merged['Tanggal'] = pd.to_datetime(future_data_merged['Tahun'].astype(str) + '-' + future_data_merged['Bulan'].astype(str) + '-01')

    # Grafik Suhu Interaktif
    st.subheader("📈 Grafik Interaktif Suhu Rata-rata")
    fig_suhu = px.line(
        future_data_merged,
        x='Tanggal',
        y='Suhu',
        color='Sumber',
        title='Suhu Rata-rata Bulanan',
        labels={'Suhu': '°C', 'Tanggal': 'Waktu'}
    )
    st.plotly_chart(fig_suhu, use_container_width=True)

    # Grafik Curah Hujan Interaktif
    st.subheader("📈 Grafik Interaktif Curah Hujan")
    fig_hujan = px.line(
        future_data_merged,
        x='Tanggal',
        y='CurahHujan',
        color='Sumber',
        title='Curah Hujan Bulanan',
        labels={'CurahHujan': 'mm', 'Tanggal': 'Waktu'}
    )
    st.plotly_chart(fig_hujan, use_container_width=True)

    # Export ke CSV
    st.subheader("💾 Simpan Hasil Prediksi")
    csv = future_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Prediksi 2025–2075",
        data=csv,
        file_name='prediksi_cuaca_2025_2075.csv',
        mime='text/csv'
    )
