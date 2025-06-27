#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import json
import datetime
from typing import Dict, List, Optional

class DatabaseManager:
    def __init__(self, db_path: str = "prediction_history.db"):
        """Initialize database manager"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        patient_age INTEGER,
                        patient_gender TEXT,
                        patient_bmi REAL,
                        patient_systolic_bp INTEGER,
                        patient_diastolic_bp INTEGER,
                        patient_smoking BOOLEAN,
                        patient_alcohol BOOLEAN,
                        patient_exercise_level INTEGER,
                        patient_family_history BOOLEAN,
                        symptoms TEXT,
                        predicted_disease TEXT,
                        confidence REAL,
                        all_probabilities TEXT,
                        model_accuracy REAL,
                        session_id TEXT
                    )
                ''')
                
                # Create user sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_predictions INTEGER DEFAULT 0
                    )
                ''')
                
                # Create disease statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS disease_statistics (
                        disease TEXT PRIMARY KEY,
                        total_predictions INTEGER DEFAULT 0,
                        correct_predictions INTEGER DEFAULT 0,
                        average_confidence REAL DEFAULT 0.0,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                print(f"Database initialized successfully: {self.db_path}")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def create_session(self, session_id: str) -> bool:
        """Create a new user session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions (session_id, last_activity)
                    VALUES (?, CURRENT_TIMESTAMP)
                ''', (session_id,))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE user_sessions 
                    SET last_activity = CURRENT_TIMESTAMP,
                        total_predictions = total_predictions + 1
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
        except Exception as e:
            print(f"Error updating session: {e}")
    
    def save_prediction(self, session_id: str, patient_data: Dict, prediction_result: Dict) -> bool:
        """Save a prediction to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract symptoms from patient data
                symptoms = []
                for key, value in patient_data.items():
                    if key.startswith('high_blood_') or key in [
                        'frequent_urination', 'excessive_thirst', 'fatigue', 'blurred_vision',
                        'headache', 'dizziness', 'chest_pain', 'shortness_of_breath',
                        'wheezing', 'coughing', 'chest_tightness', 'rapid_breathing',
                        'joint_pain', 'stiffness', 'swelling', 'reduced_range_of_motion',
                        'sadness', 'sleep_problems', 'appetite_changes', 'concentration_issues',
                        'irregular_heartbeat', 'fever', 'nausea', 'vomiting', 'diarrhea', 'constipation'
                    ]:
                        if value == 1:
                            symptoms.append(key)
                
                # Insert prediction record
                cursor.execute('''
                    INSERT INTO predictions (
                        patient_age, patient_gender, patient_bmi, patient_systolic_bp,
                        patient_diastolic_bp, patient_smoking, patient_alcohol,
                        patient_exercise_level, patient_family_history, symptoms,
                        predicted_disease, confidence, all_probabilities, model_accuracy, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient_data.get('age'),
                    patient_data.get('gender'),
                    patient_data.get('bmi'),
                    patient_data.get('systolic_bp'),
                    patient_data.get('diastolic_bp'),
                    patient_data.get('smoking'),
                    patient_data.get('alcohol_consumption'),
                    patient_data.get('exercise_level'),
                    patient_data.get('family_history'),
                    json.dumps(symptoms),
                    prediction_result.get('predicted_disease'),
                    prediction_result.get('confidence'),
                    json.dumps(prediction_result.get('all_probabilities', {})),
                    prediction_result.get('model_accuracy', 0.0),
                    session_id
                ))
                
                # Update disease statistics
                self.update_disease_statistics(prediction_result.get('predicted_disease'), 
                                            prediction_result.get('confidence'))
                
                # Update session activity
                self.update_session_activity(session_id)
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def update_disease_statistics(self, disease: str, confidence: float):
        """Update disease statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current statistics
                cursor.execute('''
                    SELECT total_predictions, average_confidence 
                    FROM disease_statistics 
                    WHERE disease = ?
                ''', (disease,))
                
                result = cursor.fetchone()
                if result:
                    total_pred, avg_conf = result
                    new_total = total_pred + 1
                    new_avg = ((avg_conf * total_pred) + confidence) / new_total
                    
                    cursor.execute('''
                        UPDATE disease_statistics 
                        SET total_predictions = ?, average_confidence = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE disease = ?
                    ''', (new_total, new_avg, disease))
                else:
                    cursor.execute('''
                        INSERT INTO disease_statistics (disease, total_predictions, average_confidence)
                        VALUES (?, 1, ?)
                    ''', (disease, confidence))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error updating disease statistics: {e}")
    
    def get_user_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get prediction history for a user session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM predictions 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (session_id, limit))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    record = dict(zip(columns, row))
                    # Parse JSON fields
                    if record['symptoms']:
                        record['symptoms'] = json.loads(record['symptoms'])
                    if record['all_probabilities']:
                        record['all_probabilities'] = json.loads(record['all_probabilities'])
                    history.append(record)
                
                return history
                
        except Exception as e:
            print(f"Error getting user history: {e}")
            return []
    
    def get_disease_statistics(self) -> Dict:
        """Get overall disease statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT disease, total_predictions, average_confidence, last_updated
                    FROM disease_statistics 
                    ORDER BY total_predictions DESC
                ''')
                
                stats = {}
                for row in cursor.fetchall():
                    disease, total, avg_conf, last_updated = row
                    stats[disease] = {
                        'total_predictions': total,
                        'average_confidence': avg_conf,
                        'last_updated': last_updated
                    }
                
                return stats
                
        except Exception as e:
            print(f"Error getting disease statistics: {e}")
            return {}
    
    def get_recent_predictions(self, limit: int = 20) -> List[Dict]:
        """Get recent predictions across all users"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                predictions = []
                for row in rows:
                    record = dict(zip(columns, row))
                    # Parse JSON fields
                    if record['symptoms']:
                        record['symptoms'] = json.loads(record['symptoms'])
                    if record['all_probabilities']:
                        record['all_probabilities'] = json.loads(record['all_probabilities'])
                    predictions.append(record)
                
                return predictions
                
        except Exception as e:
            print(f"Error getting recent predictions: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary statistics for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session info
                cursor.execute('''
                    SELECT created_at, last_activity, total_predictions
                    FROM user_sessions 
                    WHERE session_id = ?
                ''', (session_id,))
                
                session_info = cursor.fetchone()
                if not session_info:
                    return {}
                
                # Get disease distribution for this session
                cursor.execute('''
                    SELECT predicted_disease, COUNT(*) as count
                    FROM predictions 
                    WHERE session_id = ?
                    GROUP BY predicted_disease
                    ORDER BY count DESC
                ''', (session_id,))
                
                disease_distribution = dict(cursor.fetchall())
                
                return {
                    'session_id': session_id,
                    'created_at': session_info[0],
                    'last_activity': session_info[1],
                    'total_predictions': session_info[2],
                    'disease_distribution': disease_distribution
                }
                
        except Exception as e:
            print(f"Error getting session summary: {e}")
            return {}
    
    def clear_old_sessions(self, days_old: int = 30):
        """Clear old sessions and their predictions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old sessions
                cursor.execute('''
                    DELETE FROM user_sessions 
                    WHERE last_activity < datetime('now', '-{} days')
                '''.format(days_old))
                
                # Delete predictions for old sessions
                cursor.execute('''
                    DELETE FROM predictions 
                    WHERE session_id NOT IN (SELECT session_id FROM user_sessions)
                ''')
                
                conn.commit()
                print(f"Cleared sessions older than {days_old} days")
                
        except Exception as e:
            print(f"Error clearing old sessions: {e}")

def main():
    """Test the database manager"""
    db = DatabaseManager()
    
    # Test creating a session
    session_id = "test_session_123"
    db.create_session(session_id)
    
    # Test saving a prediction
    patient_data = {
        'age': 45,
        'gender': 'Male',
        'bmi': 28.5,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'smoking': 1,
        'alcohol_consumption': 0,
        'exercise_level': 1,
        'family_history': 1,
        'high_blood_sugar': 1,
        'frequent_urination': 1,
        'fatigue': 1
    }
    
    prediction_result = {
        'predicted_disease': 'Diabetes',
        'confidence': 0.85,
        'all_probabilities': {
            'Diabetes': 0.85,
            'Hypertension': 0.10,
            'Healthy': 0.05
        },
        'model_accuracy': 0.95
    }
    
    success = db.save_prediction(session_id, patient_data, prediction_result)
    print(f"Prediction saved: {success}")
    
    # Test getting history
    history = db.get_user_history(session_id)
    print(f"User history: {len(history)} records")
    
    # Test getting statistics
    stats = db.get_disease_statistics()
    print(f"Disease statistics: {stats}")

if __name__ == "__main__":
    main() 