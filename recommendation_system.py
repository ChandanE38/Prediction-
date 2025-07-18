import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import math

class RecommendationSystem:
    def __init__(self):
        self.doctors_df = None
        self.medicines_df = None
        self.disease_specialization_map = {}
        self.medicine_disease_map = {}
        self.disease_accuracy = {}
        
    def load_data(self, doctors_file, medicines_file):
        """Load doctor and medicine data"""
        print("Loading recommendation data...")
        self.doctors_df = pd.read_csv(doctors_file)
        self.medicines_df = pd.read_csv(medicines_file)
        
        # Create disease-specialization mapping
        self.disease_specialization_map = {
            'Diabetes': 'Endocrinology',
            'Hypertension': 'Cardiology',
            'Heart Disease': 'Cardiology',
            'Asthma': 'Pulmonology',
            'Arthritis': 'Orthopedics',
            'Depression': 'Psychiatry',
            'Healthy': 'General Medicine'
        }
        
        # Create medicine-disease mapping
        self.medicine_disease_map = dict(zip(self.medicines_df['name'], self.medicines_df['disease']))
        
        # Initialize disease accuracy (these would come from model evaluation)
        self.disease_accuracy = {
            'Diabetes': 0.95,
            'Hypertension': 0.92,
            'Heart Disease': 0.89,
            'Asthma': 0.94,
            'Arthritis': 0.91,
            'Depression': 0.88,
            'Healthy': 0.96
        }
        
        print(f"Doctors loaded: {len(self.doctors_df)}")
        print(f"Medicines loaded: {len(self.medicines_df)}")
        
        return self.doctors_df, self.medicines_df
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def recommend_doctors(self, predicted_disease, location=None, user_lat=None, user_lon=None, top_n=5):
        """Recommend doctors based on predicted disease and location"""
        if self.doctors_df is None:
            raise ValueError("Doctor data not loaded. Please load data first.")
        
        # Get specialization for the disease
        specialization = self.disease_specialization_map.get(predicted_disease, 'General Medicine')
        
        # Filter doctors by specialization
        filtered_doctors = self.doctors_df[
            self.doctors_df['specialization'] == specialization
        ].copy()
        
        # If no doctors found for exact specialization, look for related specializations
        if len(filtered_doctors) == 0:
            related_specializations = self._get_related_specializations(specialization)
            filtered_doctors = self.doctors_df[
                self.doctors_df['specialization'].isin(related_specializations)
            ].copy()
        
        # If location is provided, filter by location
        if location:
            filtered_doctors = filtered_doctors[
                filtered_doctors['location'].str.contains(location, case=False, na=False)
            ]
        
        # If coordinates are provided, calculate distances
        if user_lat is not None and user_lon is not None:
            # Add sample coordinates for doctors (in real system, these would be in the data)
            doctor_coordinates = {
                'New York': (40.7128, -74.0060),
                'Los Angeles': (34.0522, -118.2437),
                'Chicago': (41.8781, -87.6298),
                'Houston': (29.7604, -95.3698),
                'Phoenix': (33.4484, -112.0740),
                'Philadelphia': (39.9526, -75.1652),
                'San Antonio': (29.4241, -98.4936),
                'San Diego': (32.7157, -117.1611),
                'Dallas': (32.7767, -96.7970),
                'San Jose': (37.3382, -121.8863)
            }
            
            # Calculate distances
            distances = []
            for _, doctor in filtered_doctors.iterrows():
                if doctor['location'] in doctor_coordinates:
                    doc_lat, doc_lon = doctor_coordinates[doctor['location']]
                    distance = self.calculate_distance(user_lat, user_lon, doc_lat, doc_lon)
                    distances.append(distance)
                else:
                    distances.append(float('inf'))  # Unknown location
            
            filtered_doctors['distance_km'] = distances
            
            # Sort by distance, then by rating and experience
            filtered_doctors = filtered_doctors.sort_values(
                ['distance_km', 'rating', 'experience'], 
                ascending=[True, False, False]
            )
        else:
            # Sort by rating and experience only
            filtered_doctors = filtered_doctors.sort_values(
                ['rating', 'experience'], ascending=[False, False]
            )
        
        # Return top N doctors
        recommendations = filtered_doctors.head(top_n).to_dict('records')
        
        return recommendations
    
    def recommend_doctors_nearby(self, user_lat, user_lon, radius_km=50, specialization=None, top_n=10):
        """Find doctors within a specified radius"""
        if self.doctors_df is None:
            raise ValueError("Doctor data not loaded. Please load data first.")
        
        # Sample coordinates for doctors
        doctor_coordinates = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
            'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936),
            'San Diego': (32.7157, -117.1611),
            'Dallas': (32.7767, -96.7970),
            'San Jose': (37.3382, -121.8863)
        }
        
        # Filter by specialization if provided
        if specialization:
            filtered_doctors = self.doctors_df[
                self.doctors_df['specialization'] == specialization
            ].copy()
        else:
            filtered_doctors = self.doctors_df.copy()
        
        # Calculate distances and filter by radius
        nearby_doctors = []
        for _, doctor in filtered_doctors.iterrows():
            if doctor['location'] in doctor_coordinates:
                doc_lat, doc_lon = doctor_coordinates[doctor['location']]
                distance = self.calculate_distance(user_lat, user_lon, doc_lat, doc_lon)
                
                if distance <= radius_km:
                    doctor_dict = doctor.to_dict()
                    doctor_dict['distance_km'] = distance
                    nearby_doctors.append(doctor_dict)
        
        # Sort by distance, then by rating
        nearby_doctors.sort(key=lambda x: (x['distance_km'], -x['rating']))
        
        return nearby_doctors[:top_n]
    
    def recommend_medicines(self, predicted_disease, top_n=5):
        """Recommend medicines based on predicted disease"""
        if self.medicines_df is None:
            raise ValueError("Medicine data not loaded. Please load data first.")
        
        # Filter medicines by disease
        filtered_medicines = self.medicines_df[
            self.medicines_df['disease'] == predicted_disease
        ]
        
        # If no medicines found for exact disease, look for related diseases
        if len(filtered_medicines) == 0:
            related_diseases = self._get_related_diseases(predicted_disease)
            filtered_medicines = self.medicines_df[
                self.medicines_df['disease'].isin(related_diseases)
            ]
        
        # Return top N medicines
        recommendations = filtered_medicines.head(top_n).to_dict('records')
        
        return recommendations
    
    def get_disease_accuracy(self, disease):
        """Get accuracy for a specific disease"""
        return self.disease_accuracy.get(disease, 0.0)
    
    def get_all_disease_accuracies(self):
        """Get accuracy for all diseases"""
        return self.disease_accuracy.copy()
    
    def _get_related_specializations(self, specialization):
        """Get related medical specializations"""
        related_map = {
            'Cardiology': ['Cardiology', 'Internal Medicine'],
            'Endocrinology': ['Endocrinology', 'Internal Medicine'],
            'Neurology': ['Neurology', 'Internal Medicine'],
            'Orthopedics': ['Orthopedics', 'Physical Medicine'],
            'Psychiatry': ['Psychiatry', 'Psychology'],
            'Pulmonology': ['Pulmonology', 'Internal Medicine'],
            'General Medicine': ['General Medicine', 'Internal Medicine', 'Family Medicine']
        }
        return related_map.get(specialization, ['General Medicine'])
    
    def _get_related_diseases(self, disease):
        """Get related diseases for medicine recommendations"""
        related_map = {
            'Diabetes': ['Diabetes', 'Hypertension'],
            'Hypertension': ['Hypertension', 'Heart Disease'],
            'Heart Disease': ['Heart Disease', 'Hypertension'],
            'Asthma': ['Asthma', 'Respiratory Issues'],
            'Arthritis': ['Arthritis', 'Joint Pain'],
            'Depression': ['Depression', 'Anxiety'],
            'Healthy': ['General Health', 'Vitamins']
        }
        return related_map.get(disease, ['General Health'])
    
    def get_doctor_details(self, doctor_name):
        """Get detailed information about a specific doctor"""
        if self.doctors_df is None:
            raise ValueError("Doctor data not loaded. Please load data first.")
        
        doctor = self.doctors_df[self.doctors_df['name'] == doctor_name]
        if len(doctor) > 0:
            return doctor.iloc[0].to_dict()
        else:
            return None
    
    def get_medicine_details(self, medicine_name):
        """Get detailed information about a specific medicine"""
        if self.medicines_df is None:
            raise ValueError("Medicine data not loaded. Please load data first.")
        
        medicine = self.medicines_df[self.medicines_df['name'] == medicine_name]
        if len(medicine) > 0:
            return medicine.iloc[0].to_dict()
        else:
            return None
    
    def search_doctors(self, query, top_n=10):
        """Search doctors by name or specialization"""
        if self.doctors_df is None:
            raise ValueError("Doctor data not loaded. Please load data first.")
        
        # Search in name and specialization
        mask = (
            self.doctors_df['name'].str.contains(query, case=False, na=False) |
            self.doctors_df['specialization'].str.contains(query, case=False, na=False) |
            self.doctors_df['location'].str.contains(query, case=False, na=False)
        )
        
        results = self.doctors_df[mask].sort_values('rating', ascending=False).head(top_n)
        return results.to_dict('records')
    
    def search_medicines(self, query, top_n=10):
        """Search medicines by name or disease"""
        if self.medicines_df is None:
            raise ValueError("Medicine data not loaded. Please load data first.")
        
        # Search in name and disease
        mask = (
            self.medicines_df['name'].str.contains(query, case=False, na=False) |
            self.medicines_df['disease'].str.contains(query, case=False, na=False)
        )
        
        results = self.medicines_df[mask].head(top_n)
        return results.to_dict('records')
    
    def get_disease_info(self, disease):
        """Get information about a specific disease"""
        disease_info = {
            'Diabetes': {
                'description': 'A chronic disease that affects how your body turns food into energy.',
                'symptoms': ['High blood sugar', 'Frequent urination', 'Excessive thirst', 'Fatigue'],
                'risk_factors': ['Obesity', 'Family history', 'Age', 'Physical inactivity'],
                'prevention': ['Healthy diet', 'Regular exercise', 'Weight management', 'Regular checkups'],
                'accuracy': self.disease_accuracy.get('Diabetes', 0.0)
            },
            'Hypertension': {
                'description': 'High blood pressure that can lead to serious health problems.',
                'symptoms': ['Headaches', 'Shortness of breath', 'Nosebleeds', 'Chest pain'],
                'risk_factors': ['Age', 'Family history', 'Obesity', 'High salt intake'],
                'prevention': ['Low-sodium diet', 'Regular exercise', 'Stress management', 'Limit alcohol'],
                'accuracy': self.disease_accuracy.get('Hypertension', 0.0)
            },
            'Heart Disease': {
                'description': 'Various conditions affecting the heart and blood vessels.',
                'symptoms': ['Chest pain', 'Shortness of breath', 'Fatigue', 'Irregular heartbeat'],
                'risk_factors': ['High blood pressure', 'High cholesterol', 'Smoking', 'Diabetes'],
                'prevention': ['Healthy diet', 'Regular exercise', 'No smoking', 'Stress management'],
                'accuracy': self.disease_accuracy.get('Heart Disease', 0.0)
            },
            'Asthma': {
                'description': 'A condition that affects the airways in the lungs.',
                'symptoms': ['Wheezing', 'Shortness of breath', 'Chest tightness', 'Coughing'],
                'risk_factors': ['Family history', 'Allergies', 'Respiratory infections', 'Environmental factors'],
                'prevention': ['Avoid triggers', 'Regular medication', 'Clean environment', 'Regular checkups'],
                'accuracy': self.disease_accuracy.get('Asthma', 0.0)
            },
            'Arthritis': {
                'description': 'Inflammation of joints causing pain and stiffness.',
                'symptoms': ['Joint pain', 'Stiffness', 'Swelling', 'Reduced range of motion'],
                'risk_factors': ['Age', 'Family history', 'Previous joint injury', 'Obesity'],
                'prevention': ['Regular exercise', 'Weight management', 'Joint protection', 'Healthy diet'],
                'accuracy': self.disease_accuracy.get('Arthritis', 0.0)
            },
            'Depression': {
                'description': 'A mental health disorder characterized by persistent sadness.',
                'symptoms': ['Persistent sadness', 'Loss of interest', 'Sleep problems', 'Fatigue'],
                'risk_factors': ['Family history', 'Trauma', 'Chronic illness', 'Substance abuse'],
                'prevention': ['Social support', 'Regular exercise', 'Stress management', 'Professional help'],
                'accuracy': self.disease_accuracy.get('Depression', 0.0)
            },
            'Healthy': {
                'description': 'Good health with no significant medical conditions.',
                'symptoms': ['None'],
                'risk_factors': ['None'],
                'prevention': ['Regular exercise', 'Healthy diet', 'Regular checkups', 'Stress management'],
                'accuracy': self.disease_accuracy.get('Healthy', 0.0)
            }
        }
        
        return disease_info.get(disease, {
            'description': 'Information not available',
            'symptoms': [],
            'risk_factors': [],
            'prevention': [],
            'accuracy': 0.0
        })

def main():
    """Test the recommendation system"""
    # Initialize recommendation system
    recommender = RecommendationSystem()
    
    # Load data
    doctors_df, medicines_df = recommender.load_data('data/doctors.csv', 'data/medicines.csv')
    
    # Test doctor recommendations with location
    print("\nDoctor Recommendations for Diabetes (with location):")
    doctors = recommender.recommend_doctors('Diabetes', user_lat=40.7128, user_lon=-74.0060, top_n=3)
    for doctor in doctors:
        distance_info = f" - Distance: {doctor.get('distance_km', 'N/A'):.1f} km" if 'distance_km' in doctor else ""
        print(f"- {doctor['name']} ({doctor['specialization']}) - Rating: {doctor['rating']}{distance_info}")
    
    # Test nearby doctors
    print("\nNearby Doctors (within 100km of New York):")
    nearby_doctors = recommender.recommend_doctors_nearby(40.7128, -74.0060, radius_km=100, top_n=5)
    for doctor in nearby_doctors:
        print(f"- {doctor['name']} ({doctor['specialization']}) - {doctor['location']} - {doctor['distance_km']:.1f} km")
    
    # Test disease accuracy
    print("\nDisease Prediction Accuracies:")
    accuracies = recommender.get_all_disease_accuracies()
    for disease, accuracy in accuracies.items():
        print(f"- {disease}: {accuracy:.1%}")
    
    # Test medicine recommendations
    print("\nMedicine Recommendations for Diabetes:")
    medicines = recommender.recommend_medicines('Diabetes', top_n=3)
    for medicine in medicines:
        print(f"- {medicine['name']} ({medicine['type']}) - Side effects: {medicine['side_effects']}")
    
    # Test disease information with accuracy
    print("\nDisease Information for Diabetes:")
    disease_info = recommender.get_disease_info('Diabetes')
    print(f"Description: {disease_info['description']}")
    print(f"Accuracy: {disease_info['accuracy']:.1%}")
    print(f"Symptoms: {', '.join(disease_info['symptoms'])}")

if __name__ == "__main__":
    main()
