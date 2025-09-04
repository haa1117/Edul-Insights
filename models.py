import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import sqlite3
import joblib
import os

class EduAnalyticsML:
    def __init__(self, db_path='eduinsights.db'):
        self.db_path = db_path
        self.models = {}
        
    def load_data(self):
        """Load data from database"""
        conn = sqlite3.connect(self.db_path)
        self.students_df = pd.read_sql_query("SELECT * FROM students", conn)
        self.performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
        self.attendance_df = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()
        
        # Convert dates
        self.performance_df['assessment_date'] = pd.to_datetime(self.performance_df['assessment_date'])
        self.attendance_df['date'] = pd.to_datetime(self.attendance_df['date'])
        
    def prepare_features(self):
        """Prepare features for ML models"""
        features = []
        
        for _, student in self.students_df.iterrows():
            student_id = student['student_id']
            
            # Get student data
            perf_data = self.performance_df[self.performance_df['student_id'] == student_id]
            att_data = self.attendance_df[self.attendance_df['student_id'] == student_id]
            
            if len(perf_data) > 0:
                avg_performance = perf_data['percentage'].mean()
                attendance_rate = att_data['is_present'].mean() if len(att_data) > 0 else 0.9
                
                # Calculate risk factors
                failing_subjects = len(perf_data.groupby('subject')['percentage'].mean()[perf_data.groupby('subject')['percentage'].mean() < 50])
                
                features.append({
                    'student_id': student_id,
                    'grade': int(student['grade'].replace('th', '')),
                    'gender': 1 if student['gender'] == 'Male' else 0,
                    'family_income': {'Low': 0, 'Middle': 1, 'High': 2}[student['family_income']],
                    'motivation': student['motivation_level'],
                    'study_hours': student['study_hours_per_day'],
                    'avg_performance': avg_performance,
                    'attendance_rate': attendance_rate,
                    'failing_subjects': failing_subjects,
                    'at_risk': 1 if avg_performance < 60 or attendance_rate < 0.8 else 0,
                    'needs_intervention': 1 if avg_performance < 65 or attendance_rate < 0.8 or failing_subjects > 0 else 0
                })
        
        self.features_df = pd.DataFrame(features)
        return self.features_df
    
    def train_risk_model(self):
        """Train at-risk prediction model"""
        X = self.features_df[['grade', 'gender', 'family_income', 'motivation', 
                             'study_hours', 'avg_performance', 'attendance_rate', 'failing_subjects']]
        y = self.features_df['at_risk']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Risk prediction accuracy: {accuracy:.3f}")
        
        self.models['risk'] = model
        return model
    
    def train_performance_model(self):
        """Train performance prediction model"""
        X = self.features_df[['grade', 'motivation', 'study_hours', 'attendance_rate']]
        y = self.features_df['avg_performance']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        r2 = r2_score(y_test, model.predict(X_test))
        print(f"Performance prediction R²: {r2:.3f}")
        
        self.models['performance'] = model
        return model
    
    def get_intervention_recommendations(self, student_id):
        """Get intervention recommendations"""
        student = self.features_df[self.features_df['student_id'] == student_id].iloc[0]
        recommendations = []
        
        if student['avg_performance'] < 60:
            recommendations.append("Enroll in tutoring program")
        if student['attendance_rate'] < 0.8:
            recommendations.append("Attendance counseling")
        if student['motivation'] < 0.6:
            recommendations.append("Motivational coaching")
        if student['failing_subjects'] > 0:
            recommendations.append("Subject-specific support")
        
        return recommendations if recommendations else ["Continue current progress"]
    
    def predict_risk(self, student_id):
        """Predict student risk"""
        if 'risk' not in self.models:
            return None
        
        student = self.features_df[self.features_df['student_id'] == student_id]
        if len(student) == 0:
            return None
        
        X = student[['grade', 'gender', 'family_income', 'motivation', 
                    'study_hours', 'avg_performance', 'attendance_rate', 'failing_subjects']]
        
        risk_prob = self.models['risk'].predict_proba(X)[0][1]
        return risk_prob
    
    def get_analytics_summary(self):
        """Get comprehensive analytics"""
        return {
            'total_students': len(self.features_df),
            'at_risk_count': int(self.features_df['at_risk'].sum()),
            'at_risk_percentage': round(self.features_df['at_risk'].mean() * 100, 1),
            'avg_performance': round(self.features_df['avg_performance'].mean(), 1),
            'avg_attendance': round(self.features_df['attendance_rate'].mean() * 100, 1),
            'intervention_needed': int(self.features_df['needs_intervention'].sum())
        }
    
    def save_models(self):
        """Save trained models"""
        os.makedirs('saved_models', exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'saved_models/{name}_model.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load saved models"""
        model_files = ['risk_model.pkl', 'performance_model.pkl']
        for file in model_files:
            path = f'saved_models/{file}'
            if os.path.exists(path):
                name = file.replace('_model.pkl', '')
                self.models[name] = joblib.load(path)
        print("Models loaded successfully!")

def train_all_models():
    """Train all ML models"""
    ml = EduAnalyticsML()
    ml.load_data()
    ml.prepare_features()
    ml.train_risk_model()
    ml.train_performance_model()
    ml.save_models()
    
    summary = ml.get_analytics_summary()
    print(f"\nTraining Complete!")
    print(f"Total Students: {summary['total_students']}")
    print(f"At-Risk Students: {summary['at_risk_count']} ({summary['at_risk_percentage']}%)")
    print(f"Average Performance: {summary['avg_performance']}%")
    
    return ml

if __name__ == "__main__":
    train_all_models() 