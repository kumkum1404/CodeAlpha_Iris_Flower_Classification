import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(
    page_title="Iris Flower Classification Dashboard",
    page_icon="🌸",
    layout="wide"
)
#  CSS


st.markdown("""
<style>
.main {

    background-image: url("../images/flower_bg.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
}
 .block-container {

    background-color: rgba(255, 255, 255, 0.80);

    padding: 2rem;

    border-radius: 20px;
}           

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(
        135deg,
        #ffd6e8,
        #ffeaf4,
        #fff5fa
    );
}


            
h1 {
    font-size: 52px !important;
    font-weight: 700 !important;
    text-align: center;
    color: #6C3483;
    margin-bottom: 60px;
    padding-bottom: 10px;
}

h2 {
    font-size: 34px !important;
    font-weight: 700 !important;
    color: #1F618D;
}

h3 {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #117864;
}
            
.main-title {
    font-size: 58px;
    font-weight: 700;
    text-align: center;
    color: #2d2d44;
    margin-bottom: 70px;
    margin-top: 10px;
}

p {
    font-size: 18px !important;
    color: #2C3E50;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #D6EAF8, #EBDEF0);
    padding: 20px;
}

section[data-testid="stSidebar"] h2 {
    color: #6C3483;
}

.stButton>button {
    background-color: #e91e63;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #c7436b;
}

div[data-testid="stMetric"] {
    background-color: rgba(255, 240, 245, 0.9);
    border: 2px solid #f5b7d3;
    border-radius: 15px;
    padding: 15px;
    text-align: center;
}

[data-testid="stDataFrame"] {
    border: 2px solid #6C3483;
    border-radius: 12px;
    overflow: hidden;
    [data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.9);
}

img {
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}
}

img {
    border-radius: 15px;
    margin-top: 15px;
    margin-bottom: 20px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL

model = joblib.load(
    "../models/iris_model.pkl"
)

# LOAD DATASET

df = pd.read_csv(
    "../dataset/iris.csv"
)


st.title("🌸 Iris Flower Classification")


st.sidebar.header("Flower Measurements")

sl = st.sidebar.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.1
)

sw = st.sidebar.slider(
    "Sepal Width",
    2.0,
    5.0,
    3.5
)

pl = st.sidebar.slider(
    "Petal Length",
    1.0,
    7.0,
    1.4
)

pw = st.sidebar.slider(
    "Petal Width",
    0.1,
    3.0,
    0.2
)

# =========================================================
# PREDICTION
# =========================================================

sample = np.array([[
    sl,
    sw,
    pl,
    pw
]])

prediction = model.predict(sample)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset Rows", df.shape[0])

with col2:
    st.metric("Dataset Columns", df.shape[1])

with col3:
    st.metric("Predicted Species", prediction[0])

st.header("Dataset Overview")
st.dataframe(df.head(10))
st.header("Input Measurements")

input_df = pd.DataFrame({
    "Feature": [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ],
    "Value": [
        sl,
        sw,
        pl,
        pw
    ]
})

st.table(input_df)

# =========================================================
# PREDICTION RESULT
# =========================================================

st.header("Prediction Result")

st.success(
    f"The predicted flower species is: {prediction[0]}"
)

# =========================================================
# VISUALIZATIONS
# =========================================================

st.header("Project Visualizations")

# Pairplot
st.subheader("Pairplot Analysis")

pairplot = Image.open(
    "../outputs/pairplot.png"
)

st.image(pairplot)

# Heatmap
st.subheader("Feature Correlation Heatmap")

heatmap = Image.open(
    "../outputs/heatmap.png"
)

st.image(heatmap)

# Species Distribution
st.subheader("Species Distribution")

species_graph = Image.open(
    "../outputs/species_distribution.png"
)

st.image(species_graph)

# Accuracy Graph
st.subheader("Model Accuracy Comparison")

accuracy_graph = Image.open(
    "../outputs/accuracy_graph.png"
)

st.image(accuracy_graph)

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = Image.open(
    "../outputs/confusion_matrix.png"
)

st.image(cm)
st.markdown("---")

