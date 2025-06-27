import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_medical_data(n_samples=10000):
    """
    Generate synthetic medical data for disease prediction
    """
    np.random.seed(42)
    
    # Define diseases and their associated symptoms
    diseases = {
        'Diabetes': {
            'symptoms': ['high_blood_sugar', 'frequent_urination', 'excessive_thirst', 'fatigue', 'blurred_vision'],
            'age_range': (40, 80),
            'bmi_range': (25, 40),
            'blood_pressure_range': (130, 180)
        },
        'Hypertension': {
            'symptoms': ['high_blood_pressure', 'headache', 'dizziness', 'chest_pain', 'shortness_of_breath'],
            'age_range': (30, 70),
            'bmi_range': (25, 35),
            'blood_pressure_range': (140, 200)
        },
        'Heart Disease': {
            'symptoms': ['chest_pain', 'shortness_of_breath', 'fatigue', 'irregular_heartbeat', 'swelling'],
            'age_range': (45, 85),
            'bmi_range': (25, 40),
            'blood_pressure_range': (130, 190)
        },
        'Asthma': {
            'symptoms': ['wheezing', 'shortness_of_breath', 'coughing', 'chest_tightness', 'rapid_breathing'],
            'age_range': (5, 65),
            'bmi_range': (18, 30),
            'blood_pressure_range': (90, 140)
        },
        'Arthritis': {
            'symptoms': ['joint_pain', 'stiffness', 'swelling', 'reduced_range_of_motion', 'fatigue'],
            'age_range': (50, 85),
            'bmi_range': (20, 35),
            'blood_pressure_range': (100, 150)
        },
        'Depression': {
            'symptoms': ['sadness', 'fatigue', 'sleep_problems', 'appetite_changes', 'concentration_issues'],
            'age_range': (18, 75),
            'bmi_range': (18, 35),
            'blood_pressure_range': (90, 140)
        },
        'Healthy': {
            'symptoms': [],
            'age_range': (18, 80),
            'bmi_range': (18, 25),
            'blood_pressure_range': (90, 120)
        }
    }
    
    data = []
    
    for _ in range(n_samples):
        # Randomly select a disease
        disease = random.choice(list(diseases.keys()))
        disease_info = diseases[disease]
        
        # Generate age based on disease
        age = np.random.randint(disease_info['age_range'][0], disease_info['age_range'][1])
        
        # Generate BMI based on disease
        bmi = np.random.uniform(disease_info['bmi_range'][0], disease_info['bmi_range'][1])
        
        # Generate blood pressure based on disease
        systolic = np.random.randint(disease_info['blood_pressure_range'][0], disease_info['blood_pressure_range'][1])
        diastolic = systolic - np.random.randint(30, 50)
        
        # Generate other features
        gender = random.choice(['Male', 'Female'])
        smoking = random.choice([0, 1])
        alcohol = random.choice([0, 1])
        exercise = random.choice([0, 1, 2, 3])  # 0: none, 1: light, 2: moderate, 3: heavy
        family_history = random.choice([0, 1])
        
        # Generate symptoms based on disease
        symptoms = disease_info['symptoms'].copy()
        if disease != 'Healthy':
            # Add some random symptoms
            all_symptoms = ['fever', 'cough', 'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation']
            additional_symptoms = random.sample(all_symptoms, random.randint(0, 2))
            symptoms.extend(additional_symptoms)
        
        # Create symptom columns
        symptom_dict = {
            'high_blood_sugar': 0, 'frequent_urination': 0, 'excessive_thirst': 0, 'fatigue': 0,
            'blurred_vision': 0, 'high_blood_pressure': 0, 'headache': 0, 'dizziness': 0,
            'chest_pain': 0, 'shortness_of_breath': 0, 'wheezing': 0, 'coughing': 0,
            'chest_tightness': 0, 'rapid_breathing': 0, 'joint_pain': 0, 'stiffness': 0,
            'swelling': 0, 'reduced_range_of_motion': 0, 'sadness': 0, 'sleep_problems': 0,
            'appetite_changes': 0, 'concentration_issues': 0, 'irregular_heartbeat': 0,
            'fever': 0, 'nausea': 0, 'vomiting': 0, 'diarrhea': 0, 'constipation': 0
        }
        
        for symptom in symptoms:
            if symptom in symptom_dict:
                symptom_dict[symptom] = 1
        
        # Create patient record
        patient = {
            'age': age,
            'gender': gender,
            'bmi': round(bmi, 1),
            'systolic_bp': systolic,
            'diastolic_bp': diastolic,
            'smoking': smoking,
            'alcohol_consumption': alcohol,
            'exercise_level': exercise,
            'family_history': family_history,
            'disease': disease
        }
        
        # Add symptoms
        patient.update(symptom_dict)
        data.append(patient)
    
    return pd.DataFrame(data)

def generate_doctor_data():
    """
    Generate synthetic doctor data for recommendations
    """
    doctors = [
        {'name': 'Dr. Sarah Johnson', 'specialization': 'Cardiology', 'experience': 15, 'rating': 4.8, 'location': 'New York'},
        {'name': 'Dr. Michael Chen', 'specialization': 'Endocrinology', 'experience': 12, 'rating': 4.7, 'location': 'Los Angeles'},
        {'name': 'Dr. Emily Davis', 'specialization': 'Neurology', 'experience': 18, 'rating': 4.9, 'location': 'Chicago'},
        {'name': 'Dr. Robert Wilson', 'specialization': 'Orthopedics', 'experience': 20, 'rating': 4.6, 'location': 'Houston'},
        {'name': 'Dr. Lisa Brown', 'specialization': 'Psychiatry', 'experience': 14, 'rating': 4.8, 'location': 'Phoenix'},
        {'name': 'Dr. James Miller', 'specialization': 'Pulmonology', 'experience': 16, 'rating': 4.7, 'location': 'Philadelphia'},
        {'name': 'Dr. Maria Garcia', 'specialization': 'Cardiology', 'experience': 13, 'rating': 4.9, 'location': 'San Antonio'},
        {'name': 'Dr. David Lee', 'specialization': 'Endocrinology', 'experience': 17, 'rating': 4.6, 'location': 'San Diego'},
        {'name': 'Dr. Jennifer Taylor', 'specialization': 'Neurology', 'experience': 19, 'rating': 4.8, 'location': 'Dallas'},
        {'name': 'Dr. Christopher Anderson', 'specialization': 'Orthopedics', 'experience': 11, 'rating': 4.7, 'location': 'San Jose'}
    ]
    
    return pd.DataFrame(doctors)

def generate_medicine_data():
    """
    Generate synthetic medicine data for recommendations
    """
    medicines = [
        {'name': 'Metformin', 'disease': 'Diabetes', 'type': 'Oral', 'side_effects': 'Nausea, Diarrhea'},
        {'name': 'Insulin', 'disease': 'Diabetes', 'type': 'Injection', 'side_effects': 'Hypoglycemia'},
        {'name': 'Lisinopril', 'disease': 'Hypertension', 'type': 'Oral', 'side_effects': 'Dry cough, Dizziness'},
        {'name': 'Amlodipine', 'disease': 'Hypertension', 'type': 'Oral', 'side_effects': 'Swelling, Headache'},
        {'name': 'Aspirin', 'disease': 'Heart Disease', 'type': 'Oral', 'side_effects': 'Stomach upset'},
        {'name': 'Atorvastatin', 'disease': 'Heart Disease', 'type': 'Oral', 'side_effects': 'Muscle pain'},
        {'name': 'Albuterol', 'disease': 'Asthma', 'type': 'Inhaler', 'side_effects': 'Tremors, Rapid heartbeat'},
        {'name': 'Fluticasone', 'disease': 'Asthma', 'type': 'Inhaler', 'side_effects': 'Thrush, Hoarseness'},
        {'name': 'Ibuprofen', 'disease': 'Arthritis', 'type': 'Oral', 'side_effects': 'Stomach irritation'},
        {'name': 'Acetaminophen', 'disease': 'Arthritis', 'type': 'Oral', 'side_effects': 'Liver damage (high doses)'},
        {'name': 'Sertraline', 'disease': 'Depression', 'type': 'Oral', 'side_effects': 'Nausea, Insomnia'},
        {'name': 'Fluoxetine', 'disease': 'Depression', 'type': 'Oral', 'side_effects': 'Headache, Fatigue'}
    ]
    
    return pd.DataFrame(medicines)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate all datasets
    print("Generating synthetic medical data...")
    medical_data = generate_synthetic_medical_data(10000)
    medical_data.to_csv('data/medical_data.csv', index=False)
    
    print("Generating doctor data...")
    doctor_data = generate_doctor_data()
    doctor_data.to_csv('data/doctors.csv', index=False)
    
    print("Generating medicine data...")
    medicine_data = generate_medicine_data()
    medicine_data.to_csv('data/medicines.csv', index=False)
    
    print("Data generation completed!")
    print(f"Medical data: {len(medical_data)} records")
    print(f"Doctor data: {len(doctor_data)} records")
    print(f"Medicine data: {len(medicine_data)} records") 