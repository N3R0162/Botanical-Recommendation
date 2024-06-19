import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import joblib
import pandas as pd
import numpy as np

# Load necessary data
best_knn = joblib.load('ckpt/best_knn_model.joblib')
best_rf = joblib.load('ckpt/best_rf_model.joblib')
best_xgb = joblib.load('ckpt/best_xgb_model.joblib')
data = pd.read_csv("data/crop_data.csv")
gdf = gpd.read_file("data/stanford-bq365ww4197-geojson.json")

# Assuming these are the features used during training
training_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                     'N_P_ratio', 'K_P_ratio', 'N_K_ratio', 'temp_ph_interaction']

# Function to predict top 5 crops for a province
def predict_top_crops(province, model, data):
    province_data = data[data['province'] == province][training_features]
    predictions = model.predict_proba(province_data).mean(axis=0)
    crop_labels = data['label'].unique()
    top_crops_indices = np.argsort(predictions)[-5:][::-1]
    top_crops = crop_labels[top_crops_indices]
    return top_crops

# Ensure we have a session state to store the selected province and model
if 'selected_province' not in st.session_state:
    st.session_state.selected_province = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "KNN"

# Streamlit app
st.title("Vietnam Crop Prediction App")
st.write("Hover over a province to see its name. Click on a province to see the top 5 crops that can be grown there.")

# Map
m = folium.Map(location=[15.8700, 100.9925], zoom_start=6)

# GeoJSON Structure 
for _, row in gdf.iterrows():
    province_name = row['nam']  
    geojson_obj = {
        "type": "Feature",
        "properties": {
            "name": province_name
        },
        "geometry": row['geometry'].__geo_interface__
    }
    folium.GeoJson(
        data=geojson_obj,
        name=province_name,
        tooltip=province_name,
        style_function=lambda x: {'color': 'blue', 'fillColor': 'cyan', 'weight': 1.5},
        highlight_function=lambda x: {'weight': 3, 'color': 'yellow'},
        popup=folium.Popup(province_name),
    ).add_to(m).add_child(folium.features.GeoJsonTooltip(fields=['name']))

folium_static(m)
# Select model
model_choice = st.selectbox("Select Model", ["KNN", "Random Forest", "XGBoost"], index=["KNN", "Random Forest", "XGBoost"].index(st.session_state.selected_model))
st.session_state.selected_model = model_choice

if st.session_state.selected_province:
    province = st.session_state.selected_province
    st.write(f"Selected Province: {province}")
    
    # Select the model based on user choice
    if st.session_state.selected_model == "KNN":
        selected_model = best_knn
    elif st.session_state.selected_model == "Random Forest":
        selected_model = best_rf
    elif st.session_state.selected_model == "XGBoost":
        selected_model = best_xgb

    top_crops = predict_top_crops(province, selected_model, data)
    st.write(f"Top 5 crops for {province} using {st.session_state.selected_model}: {', '.join(top_crops)}")

st.write("If the province is not automatically selected, please enter it below:")
province = st.text_input("Selected Province", st.session_state.selected_province)
if province and province != st.session_state.selected_province:
    st.session_state.selected_province = province
    
    if st.session_state.selected_model == "KNN":
        selected_model = best_knn
    elif st.session_state.selected_model == "Random Forest":
        selected_model = best_rf
    elif st.session_state.selected_model == "XGBoost":
        selected_model = best_xgb

    top_crops = predict_top_crops(province, selected_model, data)
    st.write(f"Top 5 crops for {province} using {st.session_state.selected_model}: {', '.join(top_crops)}")
