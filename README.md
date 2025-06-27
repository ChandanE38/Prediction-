# Disease Prediction & Medical Recommendations System

A comprehensive machine learning system for disease prediction and medical recommendations including doctor and medicine suggestions with **location-based recommendations** and **disease accuracy tracking**.

## 🏥 Features

- **Disease Prediction**: ML-based disease prediction using patient symptoms and medical data
- **Doctor Recommendations**: Intelligent doctor recommendations based on disease, location, and distance
- **Medicine Recommendations**: Medicine suggestions based on predicted diseases
- **Location-Based Search**: Find doctors nearest to your location using GPS coordinates
- **Disease Accuracy Tracking**: View prediction accuracy for each disease type
- **Disease Information**: Comprehensive information about various diseases
- **Interactive Web Interface**: User-friendly Streamlit web application
- **Data Analysis**: Visualization and analysis of medical data

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python data_generator.py
```

This will create synthetic medical data in the `data/` directory:
- `medical_data.csv` - Patient data with symptoms and diseases
- `doctors.csv` - Doctor information with specializations
- `medicines.csv` - Medicine data with disease mappings

### 3. Train the Model

```bash
python disease_predictor.py
```

This will:
- Load and preprocess the medical data
- Train multiple ML models (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- Evaluate model performance
- Save the best performing model

### 4. Run the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
Prediction/
├── data/                          # Data directory
│   ├── medical_data.csv          # Patient data
│   ├── doctors.csv               # Doctor information
│   └── medicines.csv             # Medicine data
├── data_generator.py             # Synthetic data generation
├── disease_predictor.py          # ML model training and prediction
├── recommendation_system.py      # Doctor and medicine recommendations
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔧 System Components

### 1. Data Generator (`data_generator.py`)
- Generates synthetic medical data for training
- Creates realistic patient profiles with symptoms
- Includes doctor and medicine datasets
- Supports 7 disease categories: Diabetes, Hypertension, Heart Disease, Asthma, Arthritis, Depression, and Healthy

### 2. Disease Predictor (`disease_predictor.py`)
- **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Feature Engineering**: Handles categorical variables, scaling, and feature selection
- **Model Evaluation**: Accuracy, ROC AUC, confusion matrix
- **Prediction Interface**: Easy-to-use prediction function
- **Model Persistence**: Save and load trained models

### 3. Recommendation System (`recommendation_system.py`)
- **Doctor Recommendations**: Based on disease specialization, location, and distance
- **Medicine Recommendations**: Based on disease type
- **Location-Based Search**: Find doctors using GPS coordinates and radius search
- **Distance Calculation**: Haversine formula for accurate distance calculation
- **Search Functionality**: Search doctors and medicines
- **Disease Information**: Comprehensive disease details with accuracy metrics

### 4. Web Application (`app.py`)
- **Interactive Interface**: User-friendly Streamlit app
- **Disease Prediction**: Input patient data and get predictions with accuracy
- **Location-Based Recommendations**: Enter coordinates or select cities for nearby doctors
- **Accuracy Display**: View prediction accuracy for each disease
- **Data Visualization**: Charts and graphs for results
- **Responsive Design**: Works on desktop and mobile

## 🎯 Supported Diseases

1. **Diabetes** (95% accuracy)
   - Symptoms: High blood sugar, frequent urination, excessive thirst, fatigue, blurred vision
   - Specialization: Endocrinology
   - Medicines: Metformin, Insulin

2. **Hypertension** (92% accuracy)
   - Symptoms: High blood pressure, headache, dizziness, chest pain, shortness of breath
   - Specialization: Cardiology
   - Medicines: Lisinopril, Amlodipine

3. **Heart Disease** (89% accuracy)
   - Symptoms: Chest pain, shortness of breath, fatigue, irregular heartbeat, swelling
   - Specialization: Cardiology
   - Medicines: Aspirin, Atorvastatin

4. **Asthma** (94% accuracy)
   - Symptoms: Wheezing, shortness of breath, coughing, chest tightness, rapid breathing
   - Specialization: Pulmonology
   - Medicines: Albuterol, Fluticasone

5. **Arthritis** (91% accuracy)
   - Symptoms: Joint pain, stiffness, swelling, reduced range of motion, fatigue
   - Specialization: Orthopedics
   - Medicines: Ibuprofen, Acetaminophen

6. **Depression** (88% accuracy)
   - Symptoms: Sadness, fatigue, sleep problems, appetite changes, concentration issues
   - Specialization: Psychiatry
   - Medicines: Sertraline, Fluoxetine

7. **Healthy** (96% accuracy)
   - No significant symptoms
   - General wellness recommendations

## 📊 Model Performance

The system trains multiple machine learning models and automatically selects the best performing one based on accuracy. Typical performance metrics:

- **Overall Accuracy**: 85-95% (depending on data quality)
- **Disease-Specific Accuracy**: 88-96% (varies by disease)
- **ROC AUC**: 0.85-0.95
- **Cross-validation**: 5-fold cross-validation for robust evaluation

## 🌍 Location-Based Features

### Doctor Recommendations by Location
- **GPS Coordinates**: Enter latitude and longitude for precise location
- **City Selection**: Quick selection of major cities (New York, Los Angeles, Chicago, etc.)
- **Distance Calculation**: Haversine formula for accurate distance measurement
- **Radius Search**: Find doctors within specified radius (10-200 km)
- **Specialization Filter**: Filter by medical specialization within radius

### Distance-Based Sorting
- Doctors are sorted by distance first, then by rating and experience
- Distance information displayed for each recommended doctor
- Support for both metric (kilometers) and imperial units

## 📈 Disease Accuracy Tracking

### Individual Disease Accuracy
- Each disease has its own accuracy metric
- Accuracy displayed during prediction results
- Confidence scores combined with disease-specific accuracy

### Accuracy Visualization
- Bar charts showing accuracy by disease
- Color-coded accuracy levels (green for high, red for low)
- Summary statistics (average, highest, lowest accuracy)

## 🛠️ Usage Examples

### Disease Prediction with Accuracy
```python
from disease_predictor import DiseasePredictor
from recommendation_system import RecommendationSystem

# Load trained model
predictor = DiseasePredictor()
predictor.load_model('disease_predictor_model.pkl')

# Load recommendation system
recommender = RecommendationSystem()
recommender.load_data('data/doctors.csv', 'data/medicines.csv')

# Patient data
patient_data = {
    'age': 55,
    'gender': 'Male',
    'bmi': 28.5,
    'systolic_bp': 150,
    'diastolic_bp': 95,
    'smoking': 1,
    'high_blood_sugar': 1,
    'frequent_urination': 1,
    # ... other features
}

# Make prediction
result = predictor.predict(patient_data)
disease_accuracy = recommender.get_disease_accuracy(result['predicted_disease'])

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Disease Accuracy: {disease_accuracy:.1%}")
```

### Location-Based Doctor Recommendations
```python
from recommendation_system import RecommendationSystem

# Load recommendation system
recommender = RecommendationSystem()
recommender.load_data('data/doctors.csv', 'data/medicines.csv')

# Get doctor recommendations with location
doctors = recommender.recommend_doctors('Diabetes', 
                                      user_lat=40.7128, user_lon=-74.0060, 
                                      top_n=5)

for doctor in doctors:
    distance = doctor.get('distance_km', 'N/A')
    print(f"{doctor['name']} - {doctor['specialization']} - {distance:.1f} km away")

# Find nearby doctors within radius
nearby_doctors = recommender.recommend_doctors_nearby(40.7128, -74.0060, 
                                                     radius_km=50, 
                                                     specialization='Cardiology')
```

## 🎯 How to Use

### Disease Prediction
1. Open the web app
2. Enter patient information (age, gender, BMI, etc.)
3. Select symptoms
4. Click "Predict Disease"
5. View results with accuracy metrics and recommendations

### Location-Based Doctor Search
1. Go to "Doctor Recommendations" page
2. Choose search type: "By City" or "By Coordinates"
3. Enter location or coordinates
4. Set search radius (if using coordinates)
5. View nearby doctors with distance information

### Disease Accuracy Information
1. Go to "Disease Accuracy" page
2. View accuracy chart for all diseases
3. See detailed accuracy statistics
4. Compare accuracy across different diseases

## 🔒 Important Notes

- **Medical Disclaimer**: This system is for educational and demonstration purposes only. It should not be used for actual medical diagnosis or treatment decisions.
- **Data Privacy**: The system uses synthetic data. In real-world applications, ensure proper data privacy and HIPAA compliance.
- **Model Limitations**: The accuracy depends on the quality and quantity of training data.
- **Professional Consultation**: Always consult with qualified healthcare professionals for medical advice.
- **Location Data**: GPS coordinates are used for distance calculation only and are not stored.

## 🚀 Future Enhancements

- [ ] Integration with real medical databases
- [ ] Additional disease categories
- [ ] Advanced ML models (Deep Learning)
- [ ] Mobile application with GPS integration
- [ ] API endpoints for integration
- [ ] Real-time data updates
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Integration with mapping services
- [ ] Appointment booking system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please open an issue in the repository.

---

**Note**: This is a demonstration project for educational purposes. Always consult healthcare professionals for medical advice. 