# 📱 Mobile Price Range Prediction App
# Created by Heisenberg 🔥

import streamlit as st
import pandas as pd
import pickle

# ==============================
# 🔹 Load the trained model
# ==============================
@st.cache_resource
def load_model():
    with open('Best_ML.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ==============================
# 🏷️ Page Setup
# ==============================
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")

st.title("📱 Mobile Price Range Prediction App")
st.markdown("### Predict your mobile price range based on its features.")
st.sidebar.header("ℹ️ About App")
st.sidebar.info(
    """
    **Purpose:** Predict mobile price category using ML model.  
    **Categories:**  
    - 💸 0 → Low Cost  
    - 💰 1 → Medium Cost  
    - 💎 2 → High Cost  
    - 🔥 3 → Very High Cost  

    **Created by:** Heisenberg 🧠  
    """
)

# ==============================
# 🧩 Input Section
# ==============================
st.header("📊 Enter Mobile Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    battery_power = st.number_input('🔋 Battery Power (mAh)', 500, 5000, 1500)
    blue = st.selectbox('🔵 Bluetooth', ['No', 'Yes'])
    clock_speed = st.number_input('⏱️ Clock Speed (GHz)', 0.5, 3.0, 1.5)
    dual_sim = st.selectbox('📶 Dual SIM', ['No', 'Yes'])
    fc = st.number_input('🤳 Front Camera (MP)', 0, 20, 5)
    four_g = st.selectbox('🌐 4G Support', ['No', 'Yes'])
    int_memory = st.number_input('💾 Internal Memory (GB)', 2, 512, 64)

with col2:
    m_dep = st.number_input('📏 Mobile Depth (cm)', 0.1, 1.0, 0.5)
    mobile_wt = st.number_input('⚖️ Weight (grams)', 80, 300, 150)
    n_cores = st.slider('💻 Number of Cores', 1, 8, 4)
    pc = st.number_input('📷 Primary Camera (MP)', 0, 30, 13)
    px_height = st.number_input('🖼️ Pixel Height', 0, 2000, 800)
    px_width = st.number_input('🖼️ Pixel Width', 0, 2000, 1000)
    ram = st.number_input('⚡ RAM (MB)', 256, 8000, 4000)

with col3:
    sc_h = st.slider('📐 Screen Height (cm)', 5, 25, 12)
    sc_w = st.slider('📐 Screen Width (cm)', 2, 15, 7)
    talk_time = st.slider('🕑 Talk Time (hours)', 2, 30, 10)
    three_g = st.selectbox('📡 3G Support', ['No', 'Yes'])
    touch_screen = st.selectbox('📲 Touch Screen', ['No', 'Yes'])
    wifi = st.selectbox('📶 WiFi Support', ['No', 'Yes'])

# ==============================
# 🧠 Data Preparation
# ==============================
def bin_encode(x): return 1 if x == 'Yes' else 0

input_data = pd.DataFrame([[
    battery_power,
    bin_encode(blue),
    clock_speed,
    bin_encode(dual_sim),
    fc,
    bin_encode(four_g),
    int_memory,
    m_dep,
    mobile_wt,
    n_cores,
    pc,
    px_height,
    px_width,
    ram,
    sc_h,
    sc_w,
    talk_time,
    bin_encode(three_g),
    bin_encode(touch_screen),
    bin_encode(wifi)
]], columns=[
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
    'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
    'three_g', 'touch_screen', 'wifi'
])

# ==============================
# 🔮 Prediction Section
# ==============================
if st.button('🔍 Predict Price Range'):
    prediction = model.predict(input_data)[0]
    label_map = {
        0: '💸 Low Cost : 10-25k',
        1: '💰 Medium Cost : 25-50k',
        2: '💎 High Cost : 50- 100k ',
        3: '🔥 Very High Cost : 100K + '
    }

    # Display Result
    st.subheader("⚙️ Model Prediction:")
    st.success(f"Predicted Price Range: **{label_map[prediction]}**")

    # Optional: Display probability if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)
        st.write("**Class Probabilities:**")
        prob_df = pd.DataFrame(probs, columns=['Low', 'Medium', 'High', 'Very High'])
        st.bar_chart(prob_df.T)

# ==============================
# 🧾 Footer
# ==============================
st.markdown("---")
st.markdown("Made with ❤️ by **Heisenberg** using Streamlit and ML.")
