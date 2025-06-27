#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from recommendation_system import RecommendationSystem

def test_location_features():
    """Test the new location-based features"""
    print("=" * 70)
    print("🧪 TESTING LOCATION-BASED FEATURES & DISEASE ACCURACY")
    print("=" * 70)
    
    # Initialize recommendation system
    recommender = RecommendationSystem()
    
    # Load data
    doctors_df, medicines_df = recommender.load_data('data/doctors.csv', 'data/medicines.csv')
    
    print("\n📍 LOCATION-BASED DOCTOR RECOMMENDATIONS")
    print("-" * 50)
    
    # Test 1: Doctor recommendations with location (New York)
    print("\n1. Doctor recommendations for Diabetes near New York:")
    doctors = recommender.recommend_doctors('Diabetes', user_lat=40.7128, user_lon=-74.0060, top_n=3)
    for i, doctor in enumerate(doctors, 1):
        distance_info = f" - Distance: {doctor.get('distance_km', 'N/A'):.1f} km" if 'distance_km' in doctor else ""
        print(f"   {i}. {doctor['name']} ({doctor['specialization']}) - Rating: {doctor['rating']}{distance_info}")
    
    # Test 2: Nearby doctors within radius
    print("\n2. All doctors within 100km of New York:")
    nearby_doctors = recommender.recommend_doctors_nearby(40.7128, -74.0060, radius_km=100, top_n=5)
    for i, doctor in enumerate(nearby_doctors, 1):
        print(f"   {i}. {doctor['name']} ({doctor['specialization']}) - {doctor['location']} - {doctor['distance_km']:.1f} km")
    
    # Test 3: Nearby specialists
    print("\n3. Cardiologists within 200km of Los Angeles:")
    cardio_doctors = recommender.recommend_doctors_nearby(34.0522, -118.2437, radius_km=200, 
                                                        specialization='Cardiology', top_n=3)
    for i, doctor in enumerate(cardio_doctors, 1):
        print(f"   {i}. {doctor['name']} - {doctor['location']} - {doctor['distance_km']:.1f} km")
    
    print("\n📊 DISEASE PREDICTION ACCURACY")
    print("-" * 50)
    
    # Test 4: Disease accuracy information
    print("\n4. Model accuracy for each disease:")
    accuracies = recommender.get_all_disease_accuracies()
    for disease, accuracy in accuracies.items():
        print(f"   {disease}: {accuracy:.1%}")
    
    # Test 5: Individual disease accuracy
    print("\n5. Detailed accuracy for specific diseases:")
    test_diseases = ['Diabetes', 'Heart Disease', 'Asthma']
    for disease in test_diseases:
        accuracy = recommender.get_disease_accuracy(disease)
        info = recommender.get_disease_info(disease)
        print(f"   {disease}:")
        print(f"     - Model Accuracy: {accuracy:.1%}")
        print(f"     - Description: {info['description'][:100]}...")
        print(f"     - Key Symptoms: {', '.join(info['symptoms'][:3])}")
    
    # Test 6: Distance calculation
    print("\n6. Distance calculation examples:")
    distances = [
        ("New York", "Los Angeles", 40.7128, -74.0060, 34.0522, -118.2437),
        ("New York", "Chicago", 40.7128, -74.0060, 41.8781, -87.6298),
        ("Los Angeles", "San Diego", 34.0522, -118.2437, 32.7157, -117.1611)
    ]
    
    for city1, city2, lat1, lon1, lat2, lon2 in distances:
        distance = recommender.calculate_distance(lat1, lon1, lat2, lon2)
        print(f"   {city1} to {city2}: {distance:.1f} km")
    
    print("\n🎯 ENHANCED RECOMMENDATIONS")
    print("-" * 50)
    
    # Test 7: Enhanced doctor recommendations with location and accuracy
    print("\n7. Enhanced recommendations for Hypertension:")
    disease = 'Hypertension'
    accuracy = recommender.get_disease_accuracy(disease)
    doctors = recommender.recommend_doctors(disease, user_lat=29.7604, user_lon=-95.3698, top_n=2)  # Houston
    
    print(f"   Disease: {disease} (Model Accuracy: {accuracy:.1%})")
    for i, doctor in enumerate(doctors, 1):
        distance_info = f" - {doctor.get('distance_km', 'N/A'):.1f} km away" if 'distance_km' in doctor else ""
        print(f"   {i}. {doctor['name']} ({doctor['specialization']}) - Rating: {doctor['rating']}{distance_info}")
    
    # Test 8: Medicine recommendations with disease accuracy
    print("\n8. Medicine recommendations with disease accuracy:")
    medicines = recommender.recommend_medicines(disease, top_n=2)
    print(f"   Recommended medicines for {disease} (Accuracy: {accuracy:.1%}):")
    for i, medicine in enumerate(medicines, 1):
        print(f"   {i}. {medicine['name']} ({medicine['type']})")
        print(f"      Side Effects: {medicine['side_effects']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL LOCATION-BASED FEATURES TESTED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_location_features() 