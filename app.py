#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from disease_predictor import DiseasePredictor
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
        # Load disease predictor
        predictor = DiseasePredictor()
        if os.path.exists('disease_predictor_model.pkl'):
            predictor.load_model('disease_predictor_model.pkl')
        else:
            st.error("Trained model not found. Please run the training script first.")
            return None, None
        
        # Load recommendation system
        recommender = RecommendationSystem()
        if os.path.exists('data/doctors.csv') and os.path.exists('data/medicines.csv'):
            recommender.load_data('data/doctors.csv', 'data/medicines.csv')
        else:
            st.error("Recommendation data not found. Please generate data first.")
            return None, None
        
        return predictor, recommender
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Disease Prediction & Medical Recommendations</h1>', unsafe_allow_html=True)
    
    # Load models
    predictor, recommender = load_models()
    
    if predictor is None or recommender is None:
        st.error("Failed to load models. Please check the data and model files.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Disease Prediction", "Doctor Recommendations", "Medicine Recommendations", "Disease Information", "Disease Accuracy"]
    )
    
    if page == "Disease Prediction":
        show_disease_prediction(predictor, recommender)
    elif page == "Doctor Recommendations":
        show_doctor_recommendations(recommender)
    elif page == "Medicine Recommendations":
        show_medicine_recommendations(recommender)
    elif page == "Disease Information":
        show_disease_information(recommender)
    elif page == "Disease Accuracy":
        show_disease_accuracy(recommender)

def show_disease_prediction(predictor, recommender):
    """Disease prediction page"""
    st.markdown('<h2>🔍 Disease Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Enter patient information to predict potential diseases:")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.slider("Age", 1, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        
        st.subheader("Vital Signs")
        systolic_bp = st.slider("Systolic Blood Pressure", 70, 200, 120)
        diastolic_bp = st.slider("Diastolic Blood Pressure", 40, 120, 80)
        
        st.subheader("Lifestyle")
        smoking = st.checkbox("Smoking")
        alcohol = st.checkbox("Alcohol Consumption")
        exercise_level = st.selectbox("Exercise Level", 
                                    ["None", "Light", "Moderate", "Heavy"])
        family_history = st.checkbox("Family History of Disease")
    
    with col2:
        st.subheader("Symptoms")
        st.write("Select all symptoms that apply:")
        
        # Common symptoms
        symptoms = {
            'high_blood_sugar': 'High Blood Sugar',
            'frequent_urination': 'Frequent Urination',
            'excessive_thirst': 'Excessive Thirst',
            'fatigue': 'Fatigue',
            'blurred_vision': 'Blurred Vision',
            'high_blood_pressure': 'High Blood Pressure',
            'headache': 'Headache',
            'dizziness': 'Dizziness',
            'chest_pain': 'Chest Pain',
            'shortness_of_breath': 'Shortness of Breath',
            'wheezing': 'Wheezing',
            'coughing': 'Coughing',
            'chest_tightness': 'Chest Tightness',
            'rapid_breathing': 'Rapid Breathing',
            'joint_pain': 'Joint Pain',
            'stiffness': 'Stiffness',
            'swelling': 'Swelling',
            'reduced_range_of_motion': 'Reduced Range of Motion',
            'sadness': 'Persistent Sadness',
            'sleep_problems': 'Sleep Problems',
            'appetite_changes': 'Appetite Changes',
            'concentration_issues': 'Concentration Issues',
            'irregular_heartbeat': 'Irregular Heartbeat',
            'fever': 'Fever',
            'nausea': 'Nausea',
            'vomiting': 'Vomiting',
            'diarrhea': 'Diarrhea',
            'constipation': 'Constipation'
        }
        
        selected_symptoms = {}
        for key, value in symptoms.items():
            selected_symptoms[key] = st.checkbox(value)
    
    # Convert exercise level to numeric
    exercise_map = {"None": 0, "Light": 1, "Moderate": 2, "Heavy": 3}
    
    # Create patient data
    patient_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'smoking': 1 if smoking else 0,
        'alcohol_consumption': 1 if alcohol else 0,
        'exercise_level': exercise_map[exercise_level],
        'family_history': 1 if family_history else 0
    }
    
    # Add symptoms
    for symptom, value in selected_symptoms.items():
        patient_data[symptom] = 1 if value else 0
    
    # Prediction button
    if st.button("🔮 Predict Disease", type="primary"):
        with st.spinner("Analyzing patient data..."):
            try:
                # Make prediction
                result = predictor.predict(patient_data)
                
                # Display results
                st.markdown('<h3>Prediction Results</h3>', unsafe_allow_html=True)
                
                # Main prediction with accuracy
                predicted_disease = result['predicted_disease']
                disease_accuracy = recommender.get_disease_accuracy(predicted_disease)
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h4>Predicted Disease: {predicted_disease}</h4>
                    <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                    <p><strong>Model Accuracy for {predicted_disease}:</strong> {disease_accuracy:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all probabilities with accuracy
                prob_df = pd.DataFrame([
                    {'Disease': disease, 'Probability': prob, 'Accuracy': recommender.get_disease_accuracy(disease)}
                    for disease, prob in result['all_probabilities'].items()
                ]).sort_values('Probability', ascending=False)
                
                # Create visualization with accuracy
                fig = px.bar(prob_df, x='Disease', y='Probability', 
                           title='Disease Probabilities with Model Accuracy',
                           color='Accuracy',
                           color_continuous_scale='RdYlGn',
                           hover_data=['Accuracy'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Show accuracy for each disease
                st.subheader("📊 Model Accuracy for Each Disease")
                for disease, prob in result['all_probabilities'].items():
                    accuracy = recommender.get_disease_accuracy(disease)
                    st.markdown(f"""
                    <div class="accuracy-info">
                        <strong>{disease}:</strong> {prob:.1%} probability | Model Accuracy: {accuracy:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                if predicted_disease != 'Healthy':
                    st.markdown('<h4>Recommendations</h4>', unsafe_allow_html=True)
                    
                    # Location input for doctor recommendations
                    st.subheader("📍 Location for Doctor Recommendations")
                    use_location = st.checkbox("Use location-based recommendations")
                    
                    user_lat = None
                    user_lon = None
                    location = None
                    
                    if use_location:
                        col1, col2 = st.columns(2)
                        with col1:
                            user_lat = st.number_input("Latitude", value=40.7128, format="%.4f")
                        with col2:
                            user_lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
                        
                        # Common cities for quick selection
                        cities = {
                            "New York": (40.7128, -74.0060),
                            "Los Angeles": (34.0522, -118.2437),
                            "Chicago": (41.8781, -87.6298),
                            "Houston": (29.7604, -95.3698),
                            "Phoenix": (33.4484, -112.0740)
                        }
                        
                        selected_city = st.selectbox("Or select a city:", ["Custom"] + list(cities.keys()))
                        if selected_city != "Custom":
                            user_lat, user_lon = cities[selected_city]
                            location = selected_city
                    else:
                        location = st.text_input("Enter location (optional):", placeholder="e.g., New York, Los Angeles")
                    
                    # Doctor recommendations
                    doctors = recommender.recommend_doctors(predicted_disease, location=location, 
                                                          user_lat=user_lat, user_lon=user_lon, top_n=3)
                    if doctors:
                        st.subheader("👨‍⚕️ Recommended Doctors")
                        for i, doctor in enumerate(doctors, 1):
                            distance_info = ""
                            if 'distance_km' in doctor and doctor['distance_km'] != float('inf'):
                                distance_info = f"""
                                <div class="distance-info">
                                    📍 Distance: {doctor['distance_km']:.1f} km
                                </div>
                                """
                            
                            st.markdown(f"""
                            **{i}. {doctor['name']}**
                            - Specialization: {doctor['specialization']}
                            - Experience: {doctor['experience']} years
                            - Rating: {doctor['rating']}/5.0
                            - Location: {doctor['location']}
                            {distance_info}
                            """)
                            st.write("---")
                    
                    # Medicine recommendations
                    medicines = recommender.recommend_medicines(predicted_disease, top_n=3)
                    if medicines:
                        st.subheader("💊 Recommended Medicines")
                        for i, medicine in enumerate(medicines, 1):
                            st.write(f"**{i}. {medicine['name']}**")
                            st.write(f"- Type: {medicine['type']}")
                            st.write(f"- Side Effects: {medicine['side_effects']}")
                            st.write("---")
                else:
                    st.markdown("""
                    <div class="prediction-result">
                        <h4>🎉 Great News!</h4>
                        <p>Based on the provided information, you appear to be in good health. 
                        Continue maintaining a healthy lifestyle!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def show_doctor_recommendations(recommender):
    """Doctor recommendations page"""
    st.markdown('<h2>👨‍⚕️ Doctor Recommendations</h2>', unsafe_allow_html=True)
    
    # Search options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Search by Disease")
        disease = st.selectbox("Select Disease", 
                              ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression"])
        
        if st.button("Find Doctors"):
            doctors = recommender.recommend_doctors(disease, top_n=10)
            if doctors:
                st.subheader(f"Recommended Doctors for {disease}")
                for i, doctor in enumerate(doctors, 1):
                    st.write(f"**{i}. {doctor['name']}**")
                    st.write(f"- Specialization: {doctor['specialization']}")
                    st.write(f"- Experience: {doctor['experience']} years")
                    st.write(f"- Rating: {doctor['rating']}/5.0")
                    st.write(f"- Location: {doctor['location']}")
                    st.write("---")
            else:
                st.warning("No doctors found for this disease.")
    
    with col2:
        st.subheader("Search by Location")
        
        # Location-based search
        search_type = st.radio("Search type:", ["By City", "By Coordinates"])
        
        if search_type == "By City":
            location = st.text_input("Enter city name:")
            if st.button("Search by City"):
                if location:
                    doctors = recommender.search_doctors(location, top_n=10)
                    if doctors:
                        st.subheader(f"Doctors in {location}")
                        for i, doctor in enumerate(doctors, 1):
                            st.write(f"**{i}. {doctor['name']}**")
                            st.write(f"- Specialization: {doctor['specialization']}")
                            st.write(f"- Experience: {doctor['experience']} years")
                            st.write(f"- Rating: {doctor['rating']}/5.0")
                            st.write(f"- Location: {doctor['location']}")
                            st.write("---")
                    else:
                        st.warning(f"No doctors found in {location}")
                else:
                    st.warning("Please enter a location.")
        
        else:  # By Coordinates
            col_a, col_b = st.columns(2)
            with col_a:
                lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            with col_b:
                lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
            
            radius = st.slider("Search radius (km)", 10, 200, 50)
            specialization = st.selectbox("Specialization (optional)", 
                                        ["Any"] + ["Cardiology", "Endocrinology", "Neurology", "Orthopedics", "Psychiatry", "Pulmonology"])
            
            if st.button("Search by Coordinates"):
                spec = None if specialization == "Any" else specialization
                nearby_doctors = recommender.recommend_doctors_nearby(lat, lon, radius_km=radius, 
                                                                     specialization=spec, top_n=10)
                if nearby_doctors:
                    st.subheader(f"Doctors within {radius}km")
                    for i, doctor in enumerate(nearby_doctors, 1):
                        st.write(f"**{i}. {doctor['name']}**")
                        st.write(f"- Specialization: {doctor['specialization']}")
                        st.write(f"- Experience: {doctor['experience']} years")
                        st.write(f"- Rating: {doctor['rating']}/5.0")
                        st.write(f"- Location: {doctor['location']}")
                        st.write(f"- Distance: {doctor['distance_km']:.1f} km")
                        st.write("---")
                else:
                    st.warning(f"No doctors found within {radius}km of the specified location.")

def show_medicine_recommendations(recommender):
    """Medicine recommendations page"""
    st.markdown('<h2>💊 Medicine Recommendations</h2>', unsafe_allow_html=True)
    
    disease = st.selectbox("Select Disease", 
                          ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression"])
    
    if st.button("Find Medicines"):
        medicines = recommender.recommend_medicines(disease, top_n=10)
        if medicines:
            st.subheader(f"Recommended Medicines for {disease}")
            for i, medicine in enumerate(medicines, 1):
                st.write(f"**{i}. {medicine['name']}**")
                st.write(f"- Type: {medicine['type']}")
                st.write(f"- Side Effects: {medicine['side_effects']}")
                st.write("---")
        else:
            st.warning("No medicines found for this disease.")

def show_disease_information(recommender):
    """Disease information page"""
    st.markdown('<h2>📚 Disease Information</h2>', unsafe_allow_html=True)
    
    disease = st.selectbox("Select Disease", 
                          ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Arthritis", "Depression"])
    
    if disease:
        info = recommender.get_disease_info(disease)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Description")
            st.write(info['description'])
            
            st.subheader("Symptoms")
            for symptom in info['symptoms']:
                st.write(f"• {symptom}")
        
        with col2:
            st.subheader("Risk Factors")
            for factor in info['risk_factors']:
                st.write(f"• {factor}")
            
            st.subheader("Prevention")
            for prevention in info['prevention']:
                st.write(f"• {prevention}")
        
        # Show accuracy
        st.subheader("📊 Model Accuracy")
        st.markdown(f"""
        <div class="accuracy-info">
            <strong>Prediction Accuracy for {disease}:</strong> {info['accuracy']:.1%}
        </div>
        """, unsafe_allow_html=True)

def show_disease_accuracy(recommender):
    """Disease accuracy page"""
    st.markdown('<h2>📊 Disease Prediction Accuracy</h2>', unsafe_allow_html=True)
    
    # Get all disease accuracies
    accuracies = recommender.get_all_disease_accuracies()
    
    # Create accuracy chart
    accuracy_df = pd.DataFrame([
        {'Disease': disease, 'Accuracy': accuracy}
        for disease, accuracy in accuracies.items()
    ]).sort_values('Accuracy', ascending=False)
    
    # Display chart
    fig = px.bar(accuracy_df, x='Disease', y='Accuracy', 
                title='Model Accuracy by Disease',
                color='Accuracy',
                color_continuous_scale='RdYlGn')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed accuracy information
    st.subheader("Detailed Accuracy Information")
    for disease, accuracy in accuracies.items():
        st.markdown(f"""
        <div class="accuracy-info">
            <strong>{disease}:</strong> {accuracy:.1%} accuracy
        </div>
        """, unsafe_allow_html=True)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    avg_accuracy = np.mean(list(accuracies.values()))
    max_accuracy = max(accuracies.values())
    min_accuracy = min(accuracies.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
    with col2:
        st.metric("Highest Accuracy", f"{max_accuracy:.1%}")
    with col3:
        st.metric("Lowest Accuracy", f"{min_accuracy:.1%}")

if __name__ == "__main__":
    main() 