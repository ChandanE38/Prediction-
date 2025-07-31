#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from recommendation_system import RecommendationSystem

# Page configuration
st.set_page_config(
    page_title="Disease Prediction & Medical Recommendations",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .accuracy-info {
        background-color: #f0f8ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .distance-info {
        background-color: #fff3cd;
        padding: 0.3rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and data"""
    try:
        # Load recommendation system
        recommender = RecommendationSystem()
        if os.path.exists('data/doctors.csv') and os.path.exists('data/medicines.csv'):
            recommender.load_data('data/doctors.csv', 'data/medicines.csv')
        else:
            st.error("Recommendation data not found. Please generate data first.")
            return None
        
        return recommender
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def simple_predict(patient_data):
    """Simple prediction function using existing model"""
    try:
        # Try to load existing model
        if os.path.exists('test_model.pkl'):
            model_data = joblib.load('test_model.pkl')
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']
            feature_columns = model_data['feature_columns']
            
            # Prepare features
            features = []
            for col in feature_columns:
                if col in patient_data:
                    if col == 'gender':
                        features.append(label_encoder.transform([patient_data[col]])[0])
                    else:
                        features.append(patient_data[col])
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            confidence = max(model.predict_proba(X_scaled)[0])
            
            return {
                "predicted_disease": prediction,
                "confidence": confidence
            }
        else:
            return {"error": "No trained model found"}
            
    except Exception as e:
        return {"error": f"Prediction error: {e}"}

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Disease Prediction & Medical Recommendations</h1>', unsafe_allow_html=True)
    
    # Load models
    recommender = load_models()
    
    if recommender is None:
        st.error("Failed to load models. Please check the data and model files.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Disease Prediction", "Doctor Recommendations", "Medicine Recommendations", "Disease Information", "Disease Accuracy"]
    )
    
    if page == "Disease Prediction":
        show_disease_prediction(recommender)
    elif page == "Doctor Recommendations":
        show_doctor_recommendations(recommender)
    elif page == "Medicine Recommendations":
        show_medicine_recommendations(recommender)
    elif page == "Disease Information":
        show_disease_information(recommender)
    elif page == "Disease Accuracy":
        show_disease_accuracy(recommender)

def show_disease_prediction(recommender):
    """Show disease prediction interface"""
    st.header("🔬 Disease Prediction")
    
    # Patient information form
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)
            smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col2:
            st.subheader("Symptoms")
            high_blood_sugar = st.checkbox("High Blood Sugar")
            frequent_urination = st.checkbox("Frequent Urination")
            excessive_thirst = st.checkbox("Excessive Thirst")
            fatigue = st.checkbox("Fatigue")
            blurred_vision = st.checkbox("Blurred Vision")
            headache = st.checkbox("Headache")
            dizziness = st.checkbox("Dizziness")
            chest_pain = st.checkbox("Chest Pain")
            shortness_of_breath = st.checkbox("Shortness of Breath")
            irregular_heartbeat = st.checkbox("Irregular Heartbeat")
            swelling = st.checkbox("Swelling")
            wheezing = st.checkbox("Wheezing")
            coughing = st.checkbox("Coughing")
            chest_tightness = st.checkbox("Chest Tightness")
            rapid_breathing = st.checkbox("Rapid Breathing")
            joint_pain = st.checkbox("Joint Pain")
            stiffness = st.checkbox("Stiffness")
            reduced_range_of_motion = st.checkbox("Reduced Range of Motion")
            sadness = st.checkbox("Sadness")
            sleep_problems = st.checkbox("Sleep Problems")
            appetite_changes = st.checkbox("Appetite Changes")
            concentration_issues = st.checkbox("Concentration Issues")
        
        submitted = st.form_submit_button("Predict Disease")
        
        if submitted:
            # Prepare patient data
            patient_data = {
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'smoking': smoking,
                'high_blood_sugar': 1 if high_blood_sugar else 0,
                'frequent_urination': 1 if frequent_urination else 0,
                'excessive_thirst': 1 if excessive_thirst else 0,
                'fatigue': 1 if fatigue else 0,
                'blurred_vision': 1 if blurred_vision else 0,
                'headache': 1 if headache else 0,
                'dizziness': 1 if dizziness else 0,
                'chest_pain': 1 if chest_pain else 0,
                'shortness_of_breath': 1 if shortness_of_breath else 0,
                'irregular_heartbeat': 1 if irregular_heartbeat else 0,
                'swelling': 1 if swelling else 0,
                'wheezing': 1 if wheezing else 0,
                'coughing': 1 if coughing else 0,
                'chest_tightness': 1 if chest_tightness else 0,
                'rapid_breathing': 1 if rapid_breathing else 0,
                'joint_pain': 1 if joint_pain else 0,
                'stiffness': 1 if stiffness else 0,
                'reduced_range_of_motion': 1 if reduced_range_of_motion else 0,
                'sadness': 1 if sadness else 0,
                'sleep_problems': 1 if sleep_problems else 0,
                'appetite_changes': 1 if appetite_changes else 0,
                'concentration_issues': 1 if concentration_issues else 0
            }
            
            # Make prediction
            result = simple_predict(patient_data)
            
            if "error" not in result:
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.success(f"**Predicted Disease:** {result['predicted_disease']}")
                st.info(f"**Confidence:** {result['confidence']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Get disease accuracy
                disease_accuracy = recommender.get_disease_accuracy(result['predicted_disease'])
                st.markdown(f'<div class="accuracy-info">**Disease Accuracy:** {disease_accuracy:.1%}</div>', unsafe_allow_html=True)
                
                # Show recommendations
                st.subheader("💊 Recommended Medicines")
                medicines = recommender.recommend_medicines(result['predicted_disease'], top_n=3)
                for i, medicine in enumerate(medicines, 1):
                    st.write(f"{i}. **{medicine['name']}** - Type: {medicine['type']}")
                
                st.subheader("👨‍⚕️ Recommended Doctors")
                doctors = recommender.recommend_doctors(result['predicted_disease'], top_n=3)
                for i, doctor in enumerate(doctors, 1):
                    st.write(f"{i}. **Dr. {doctor['name']}** - {doctor['specialization']} (Rating: {doctor['rating']}/5)")
            else:
                st.error(f"Prediction failed: {result['error']}")

def show_doctor_recommendations(recommender):
    """Show doctor recommendations interface"""
    st.header("👨‍⚕️ Doctor Recommendations")
    
    # Search options
    search_type = st.radio("Search by:", ["By Disease", "By Location"])
    
    if search_type == "By Disease":
        disease = st.selectbox("Select Disease:", ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression", "Healthy"])
        doctors = recommender.recommend_doctors(disease, top_n=5)
        
        st.subheader(f"Recommended Doctors for {disease}")
        for i, doctor in enumerate(doctors, 1):
            with st.expander(f"{i}. Dr. {doctor['name']} - {doctor['specialization']}"):
                st.write(f"**Location:** {doctor['location']}")
                st.write(f"**Rating:** {doctor['rating']}/5")
                st.write(f"**Experience:** {doctor['experience']} years")
                st.write(f"**Specialization:** {doctor['specialization']}")
    
    else:
        st.subheader("Location-Based Search")
        col1, col2 = st.columns(2)
        
        with col1:
            lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
        
        with col2:
            radius = st.slider("Search Radius (km)", 10, 200, 50)
            specialization = st.selectbox("Specialization (Optional):", ["Any"] + ["Cardiology", "Endocrinology", "Pulmonology", "Orthopedics", "Psychiatry"])
        
        if st.button("Find Nearby Doctors"):
            if specialization == "Any":
                doctors = recommender.recommend_doctors_nearby(lat, lon, radius_km=radius)
            else:
                doctors = recommender.recommend_doctors_nearby(lat, lon, radius_km=radius, specialization=specialization)
            
            if doctors:
                st.subheader(f"Doctors within {radius} km")
                for i, doctor in enumerate(doctors, 1):
                    distance = doctor.get('distance_km', 'N/A')
                    with st.expander(f"{i}. Dr. {doctor['name']} - {distance:.1f} km away"):
                        st.write(f"**Specialization:** {doctor['specialization']}")
                        st.write(f"**Location:** {doctor['location']}")
                        st.write(f"**Rating:** {doctor['rating']}/5")
                        st.write(f"**Experience:** {doctor['experience']} years")
            else:
                st.warning("No doctors found in the specified radius.")

def show_medicine_recommendations(recommender):
    """Show medicine recommendations interface"""
    st.header("💊 Medicine Recommendations")
    
    disease = st.selectbox("Select Disease:", ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression", "Healthy"])
    
    medicines = recommender.recommend_medicines(disease, top_n=10)
    
    st.subheader(f"Recommended Medicines for {disease}")
    for i, medicine in enumerate(medicines, 1):
        with st.expander(f"{i}. {medicine['name']}"):
            st.write(f"**Type:** {medicine['type']}")
            st.write(f"**Disease:** {medicine['disease']}")
            st.write(f"**Side Effects:** {medicine['side_effects']}")

def show_disease_information(recommender):
    """Show disease information"""
    st.header("📋 Disease Information")
    
    disease = st.selectbox("Select Disease:", ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression", "Healthy"])
    
    disease_info = recommender.get_disease_info(disease)
    
    if disease_info:
        st.subheader(disease)
        st.write(f"**Accuracy:** {disease_info['accuracy']:.1%}")
        st.write(f"**Specialization:** {disease_info['specialization']}")
        st.write(f"**Common Symptoms:** {disease_info['symptoms']}")
        st.write(f"**Description:** {disease_info['description']}")

def show_disease_accuracy(recommender):
    """Show disease accuracy information"""
    st.header("📊 Disease Accuracy Statistics")
    
    # Get accuracy data for all diseases
    diseases = ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression", "Healthy"]
    accuracies = [recommender.get_disease_accuracy(disease) for disease in diseases]
    
    # Create accuracy chart
    fig = px.bar(
        x=diseases,
        y=accuracies,
        title="Disease Prediction Accuracy",
        labels={'x': 'Disease', 'y': 'Accuracy (%)'},
        color=accuracies,
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy table
    st.subheader("Detailed Accuracy Information")
    accuracy_data = pd.DataFrame({
        'Disease': diseases,
        'Accuracy (%)': [f"{acc:.1f}%" for acc in accuracies]
    })
    st.table(accuracy_data)

if __name__ == "__main__":
    main() 