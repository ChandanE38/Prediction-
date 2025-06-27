#!/usr/bin/env python3
"""
Main script to run the Disease Prediction & Medical Recommendations System
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print system banner"""
    print("=" * 80)
    print("🏥 DISEASE PREDICTION & MEDICAL RECOMMENDATIONS SYSTEM")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    try:
        import pandas
        import numpy
        import sklearn
        import streamlit
        import plotly
        import joblib
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False

def generate_data():
    """Generate synthetic medical data"""
    print("📊 Generating synthetic medical data...")
    try:
        from data_generator import generate_synthetic_medical_data, generate_doctor_data, generate_medicine_data
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Generate data
        medical_data = generate_synthetic_medical_data(10000)
        medical_data.to_csv('data/medical_data.csv', index=False)
        
        doctor_data = generate_doctor_data()
        doctor_data.to_csv('data/doctors.csv', index=False)
        
        medicine_data = generate_medicine_data()
        medicine_data.to_csv('data/medicines.csv', index=False)
        
        print(f"✅ Generated {len(medical_data)} patient records")
        print(f"✅ Generated {len(doctor_data)} doctor records")
        print(f"✅ Generated {len(medicine_data)} medicine records")
        return True
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False

def train_model():
    """Train the disease prediction model"""
    print("🤖 Training disease prediction model...")
    try:
        from disease_predictor import DiseasePredictor
        
        # Initialize and train predictor
        predictor = DiseasePredictor()
        data = predictor.load_data('data/medical_data.csv')
        X_train, X_test, y_train, y_test = predictor.preprocess_data()
        models = predictor.train_models()
        predictor.save_model()
        
        print("✅ Model training completed!")
        return True
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def test_recommendations():
    """Test the recommendation system"""
    print("🔍 Testing recommendation system...")
    try:
        from recommendation_system import RecommendationSystem
        
        # Test recommendation system
        recommender = RecommendationSystem()
        doctors_df, medicines_df = recommender.load_data('data/doctors.csv', 'data/medicines.csv')
        
        # Test doctor recommendations
        doctors = recommender.recommend_doctors('Diabetes', top_n=3)
        print(f"✅ Found {len(doctors)} doctor recommendations for Diabetes")
        
        # Test medicine recommendations
        medicines = recommender.recommend_medicines('Diabetes', top_n=3)
        print(f"✅ Found {len(medicines)} medicine recommendations for Diabetes")
        
        return True
    except Exception as e:
        print(f"❌ Error testing recommendations: {e}")
        return False

def run_web_app():
    """Run the Streamlit web application"""
    print("🌐 Starting web application...")
    print("The application will open in your default browser.")
    print("Press Ctrl+C to stop the application.")
    print()
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running web app: {e}")

def main():
    """Main function to run the entire system"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("🚀 Starting Disease Prediction & Medical Recommendations System")
    print()
    
    # Check if data exists
    if not os.path.exists('data/medical_data.csv'):
        print("📊 Data not found. Generating synthetic data...")
        if not generate_data():
            print("❌ Failed to generate data. Exiting.")
            return
    else:
        print("✅ Data already exists.")
    
    # Check if model exists
    if not os.path.exists('disease_predictor_model.pkl'):
        print("🤖 Trained model not found. Training model...")
        if not train_model():
            print("❌ Failed to train model. Exiting.")
            return
    else:
        print("✅ Trained model already exists.")
    
    # Test recommendations
    if not test_recommendations():
        print("❌ Failed to test recommendations. Exiting.")
        return
    
    print()
    print("🎉 System is ready!")
    print()
    
    # Ask user what to do next
    while True:
        print("What would you like to do?")
        print("1. Run the web application")
        print("2. Regenerate data")
        print("3. Retrain model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_web_app()
            break
        elif choice == '2':
            if generate_data():
                print("✅ Data regenerated successfully!")
            else:
                print("❌ Failed to regenerate data.")
        elif choice == '3':
            if train_model():
                print("✅ Model retrained successfully!")
            else:
                print("❌ Failed to retrain model.")
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")
        
        print()

if __name__ == "__main__":
    main() 