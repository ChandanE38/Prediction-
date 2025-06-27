#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def test_data_generation():
    """Test data generation"""
    print("Testing data generation...")
    try:
        # Load data
        medical_data = pd.read_csv('data/medical_data.csv')
        doctors_data = pd.read_csv('data/doctors.csv')
        medicines_data = pd.read_csv('data/medicines.csv')
        
        print(f"✅ Medical data: {len(medical_data)} records")
        print(f"✅ Doctors data: {len(doctors_data)} records")
        print(f"✅ Medicines data: {len(medicines_data)} records")
        
        # Check data structure
        print(f"Medical data columns: {list(medical_data.columns[:10])}...")
        print(f"Diseases: {medical_data['disease'].unique()}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_training():
    """Test model training"""
    print("\nTesting model training...")
    try:
        # Load data
        data = pd.read_csv('data/medical_data.csv')
        
        # Encode gender
        le = LabelEncoder()
        data['gender_encoded'] = le.fit_transform(data['gender'])
        
        # Select features
        feature_columns = [
            'age', 'gender_encoded', 'bmi', 'systolic_bp', 'diastolic_bp',
            'smoking', 'alcohol_consumption', 'exercise_level', 'family_history',
            'high_blood_sugar', 'frequent_urination', 'excessive_thirst', 'fatigue',
            'blurred_vision', 'high_blood_pressure', 'headache', 'dizziness',
            'chest_pain', 'shortness_of_breath', 'wheezing', 'coughing',
            'chest_tightness', 'rapid_breathing', 'joint_pain', 'stiffness',
            'swelling', 'reduced_range_of_motion', 'sadness', 'sleep_problems',
            'appetite_changes', 'concentration_issues', 'irregular_heartbeat',
            'fever', 'nausea', 'vomiting', 'diarrhea', 'constipation'
        ]
        
        X = data[feature_columns]
        y = data['disease']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        print(f"✅ Model trained successfully!")
        print(f"✅ Accuracy: {accuracy:.4f}")
        
        # Save model
        model_data = {
            'model': model,
            'label_encoder': le,
            'feature_columns': feature_columns
        }
        joblib.dump(model_data, 'test_model.pkl')
        print("✅ Model saved to test_model.pkl")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction():
    """Test prediction"""
    print("\nTesting prediction...")
    try:
        # Load model
        model_data = joblib.load('test_model.pkl')
        model = model_data['model']
        le = model_data['label_encoder']
        feature_columns = model_data['feature_columns']
        
        # Test patient
        test_patient = {
            'age': 55,
            'gender': 'Male',
            'bmi': 28.5,
            'systolic_bp': 150,
            'diastolic_bp': 95,
            'smoking': 1,
            'alcohol_consumption': 0,
            'exercise_level': 1,
            'family_history': 1,
            'high_blood_sugar': 1,
            'frequent_urination': 1,
            'excessive_thirst': 1,
            'fatigue': 1,
            'blurred_vision': 0,
            'high_blood_pressure': 0,
            'headache': 0,
            'dizziness': 0,
            'chest_pain': 0,
            'shortness_of_breath': 0,
            'wheezing': 0,
            'coughing': 0,
            'chest_tightness': 0,
            'rapid_breathing': 0,
            'joint_pain': 0,
            'stiffness': 0,
            'swelling': 0,
            'reduced_range_of_motion': 0,
            'sadness': 0,
            'sleep_problems': 0,
            'appetite_changes': 0,
            'concentration_issues': 0,
            'irregular_heartbeat': 0,
            'fever': 0,
            'nausea': 0,
            'vomiting': 0,
            'diarrhea': 0,
            'constipation': 0
        }
        
        # Encode gender
        test_patient['gender_encoded'] = le.transform([test_patient['gender']])[0]
        
        # Create feature vector
        features = [test_patient[col] for col in feature_columns]
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        print(f"✅ Prediction successful!")
        print(f"✅ Predicted Disease: {prediction}")
        print(f"✅ Confidence: {max(probabilities):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_recommendations():
    """Test recommendation system"""
    print("\nTesting recommendation system...")
    try:
        # Load data
        doctors_data = pd.read_csv('data/doctors.csv')
        medicines_data = pd.read_csv('data/medicines.csv')
        
        # Test doctor recommendations
        diabetes_doctors = doctors_data[doctors_data['specialization'] == 'Endocrinology']
        print(f"✅ Found {len(diabetes_doctors)} endocrinologists")
        
        # Test medicine recommendations
        diabetes_medicines = medicines_data[medicines_data['disease'] == 'Diabetes']
        print(f"✅ Found {len(diabetes_medicines)} diabetes medicines")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 TESTING DISEASE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Run tests
    tests = [
        test_data_generation,
        test_model_training,
        test_prediction,
        test_recommendations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 