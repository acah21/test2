# ===============================
# 1️⃣ Import Library
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import pydeck as pdk
import webbrowser

# ===============================
# 2️⃣ Load Dataset & Models
# ===============================
df = pd.read_csv("dataset_gunung_preprocessed.csv")

# Preprocessing
le_diff = joblib.load("le_diff.pkl")
le_level = joblib.load("le_level.pkl")
scaler = joblib.load("scaler.pkl")

# Load MLP tanpa compile
mlp_model = load_model("mlp_model_gunung.h5", compile=False)

# ===============================
# 3️⃣ Sidebar Input User
# ===============================
st.sidebar.header("Pilih Preferensi Pendakian")

province = st.sidebar.selectbox("Provinsi:", options=df['Province'].unique())
difficulty = st.sidebar.selectbox("Tingkat Kesulitan:", options=df['difficulty_level'].unique())
duration = st.sidebar.slider("Durasi Pendakian (jam):", 0, 12, 4)
level = st.sidebar.selectbox("Level Pendaki:", options=df['recommended_for'].unique())
max_distance = st.sidebar.slider("Jarak Maksimal (km):", 0, 50, 10)

# Tombol submit di sidebar
submit_button = st.sidebar.button("Tampilkan Rekomendasi")

# ===============================
# 4️⃣ Fungsi Content-Based Filtering
# ===============================
def content_based_recommendation(user_input, df, top_n=5):
    user_scaled = scaler.transform([[user_input['elevation_m'],
                                     user_input['hiking_duration_hours'],
                                     user_input['distance_km'],
                                     user_input['Elevation_gain']]])[0]

    user_vector = np.array([[ 
        user_scaled[0],
        user_scaled[1],
        user_scaled[2],
        user_scaled[3],
        le_diff.transform([user_input['difficulty_level']])[0],
        le_level.transform([user_input['recommended_for']])[0]
    ]])
    
    features = ['elevation_scaled','duration_scaled','distance_scaled','gain_scaled','difficulty_encoded','level_encoded']
    
    similarity = cosine_similarity(user_vector, df[features])
    
    df_result = df.copy()
    df_result['similarity'] = similarity[0]
    
    top_gunung = df_result.sort_values('similarity', ascending=False).head(top_n)
    return top_gunung

# ===============================
# 5️⃣ Proses setelah klik tombol
# ===============================
if submit_button:
    user_input = {
        'elevation_m': df['elevation_m'].median(),
        'hiking_duration_hours': duration,
        'distance_km': max_distance,
        'Elevation_gain': df['Elevation_gain'].median(),
        'difficulty_level': difficulty,
        'recommended_for': level
    }

    # Hitung Top Kandidat CBF
    candidate_gunung = content_based_recommendation(user_input, df, top_n=20)

    # Hitung Skor MLP
    features = ['elevation_scaled','duration_scaled','distance_scaled','gain_scaled','difficulty_encoded','level_encoded']
    candidate_gunung['mlp_score'] = mlp_model.predict(candidate_gunung[features]).flatten()

    # Ranking berdasarkan MLP
    top_gunung = candidate_gunung.sort_values('mlp_score', ascending=False).head(5)

    # ===============================
    # 6️⃣ Tampilkan Rekomendasi
    # ===============================
    st.title("🏔 Rekomendasi Gunung di Pulau Jawa")

    for idx, row in top_gunung.iterrows():
        st.markdown(f"### {row['Name']}")
        st.write(f"**Provinsi:** {row['Province']}")
        st.write(f"**Elevation:** {row['elevation_m']} m")
        st.write(f"**Difficulty:** {row['difficulty_level']}")
        st.write(f"**Hiking Duration:** {row['hiking_duration_hours']:.2f} jam")
        st.write(f"**Recommended for:** {row['recommended_for']}")
        st.image(row['image_url'], width=500)  # gambar lebih besar

        maps_url = f"https://www.google.com/maps/search/?api=1&query={row['Latitude']},{row['Longitude']}"
        if st.button(f"Buka {row['Name']} di Google Maps", key=idx):
            webbrowser.open_new_tab(maps_url)

    # ===============================
    # 7️⃣ Map Interaktif
    # ===============================
    st.subheader("📍 Lokasi Gunung Rekomendasi di Peta")
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=top_gunung,
        get_position='[Longitude, Latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius=5000,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=top_gunung['Latitude'].mean(),
        longitude=top_gunung['Longitude'].mean(),
        zoom=6,
        pitch=0
    )

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/streets-v11',
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "text": "{Name}\nProvinsi: {Province}\nElevation: {elevation_m} m\nDifficulty: {difficulty_level}\nDuration: {hiking_duration_hours:.2f} jam"
        }
    )

    st.pydeck_chart(deck)
