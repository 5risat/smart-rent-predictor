import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد التصميم والخلفية
def load_custom_css():
    css = """
    <style>
        .main {
            background-image: url("https://images.unsplash.com/photo-1581091012184-5c46a72c48b1");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .title-custom {
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-bottom: 20px;
        }
        .logo {
            font-size: 18px;
            text-align: center;
            color: #ffffff;
            font-style: italic;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# إعداد الصفحة
st.set_page_config(page_title="Smart Rent Predictor", layout="centered")
load_custom_css()
st.markdown('<div class="title-custom">🏡 Smart Rent Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="logo">Developed by Mohammed Khrisat</div>', unsafe_allow_html=True)

st.markdown("This app allows you to either upload your own dataset or enter the data manually to train a simple linear regression model to predict rent based on area.")

# اختيار طريقة إدخال البيانات
data_input_method = st.radio("📌 Select data input method:", ["Upload CSV file", "Enter data manually"])

df = None

if data_input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("📤 Upload a CSV file with 'area' and 'rent' columns", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'area' not in df.columns or 'rent' not in df.columns:
                st.error("❌ The file must contain 'area' and 'rent' columns.")
                df = None
            else:
                st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading the file: {e}")

elif data_input_method == "Enter data manually":
    st.info("⬇️ Enter the data manually below")
    data = {
        "area": [50, 60, 70],
        "rent": [200, 250, 300]
    }
    df = st.data_editor(pd.DataFrame(data), num_rows="dynamic", use_container_width=True)

# تنفيذ النموذج إذا توفرت البيانات
if df is not None:
    if len(df) < 2:
        st.warning("⚠️ Please provide at least two rows of data.")
    else:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())

        st.subheader("📊 Statistical Summary")
        st.write(df.describe())

        # إعداد الخط للغة الإنجليزية
        plt.rcParams['font.family'] = 'Arial'

        st.subheader("🧮 Rent vs. Area")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x='area', y='rent', ax=ax1)
        ax1.set_xlabel("Area (m²)")
        ax1.set_ylabel("Rent (JOD)")
        ax1.grid(True)
        st.pyplot(fig1)

        # تدريب النموذج
        model = LinearRegression()
        X = df[['area']]
        y = df['rent']
        model.fit(X, y)

        st.subheader("🔧 Model Info")
        st.write(f"📐 Coefficient: {model.coef_[0]:.2f}")
        st.write(f"📈 Intercept: {model.intercept_:.2f}")

        # التوقع
        st.subheader("💡 Predict Rent")
        input_area = st.number_input("Enter apartment area (m²)", min_value=10.0, max_value=1000.0, value=100.0, step=5.0)
        predicted_rent = model.predict([[input_area]])[0]
        st.success(f"💰 Predicted rent for {input_area} m²: **{predicted_rent:.2f} JOD**")

        st.subheader("📈 Regression Plot")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['area'], df['rent'], label='Data', color='blue')
        ax2.plot(df['area'], model.predict(X), color='red', label='Regression Line')
        ax2.scatter(input_area, predicted_rent, color='green', s=100, label='Prediction')
        ax2.set_xlabel("Area (m²)")
        ax2.set_ylabel("Rent (JOD)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)


#Made By Mohammed Khrisat
