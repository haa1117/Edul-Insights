import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import sqlite3
import joblib
import warnings
warnings.filterwarnings('ignore')

class EduAnalyticsML:
    def __init__(self, db_path='eduinsights.db'):
        self.db_path = db_path
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        
    def load_data(self):
        """Load and prepare data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load all tables
        self.students_df = pd.read_sql_query("SELECT * FROM students", conn)
        self.teachers_df = pd.read_sql_query("SELECT * FROM teachers", conn)
        self.performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
        self.attendance_df = pd.read_sql_query("SELECT * FROM attendance", conn)
        
        conn.close()
        
        # Convert date columns
        self.performance_df['assessment_date'] = pd.to_datetime(self.performance_df['assessment_date'])
        self.attendance_df['date'] = pd.to_datetime(self.attendance_df['date'])
        
        print("Data loaded successfully!")
        print(f"Students: {len(self.students_df)}")
        print(f"Performance records: {len(self.performance_df)}")
        print(f"Attendance records: {len(self.attendance_df)}")
        
    def prepare_features(self):
        """Prepare feature sets for ML models"""
        
        # Calculate student aggregated metrics
        student_features = []
        
        for _, student in self.students_df.iterrows():
            student_id = student['student_id']
            
            # Performance metrics
            student_performance = self.performance_df[self.performance_df['student_id'] == student_id]
            student_attendance = self.attendance_df[self.attendance_df['student_id'] == student_id]
            
            if len(student_performance) > 0 and len(student_attendance) > 0:
                # Academic metrics
                avg_percentage = student_performance['percentage'].mean()
                std_percentage = student_performance['percentage'].std()
                
                # Subject-wise performance
                subject_performance = student_performance.groupby('subject')['percentage'].mean()
                math_performance = subject_performance.get('Mathematics', avg_percentage)
                science_performance = subject_performance.get('Science', avg_percentage)
                english_performance = subject_performance.get('English', avg_percentage)
                
                # Assessment type performance
                assessment_performance = student_performance.groupby('assessment_type')['percentage'].mean()
                quiz_avg = assessment_performance.get('Quiz', avg_percentage)
                assignment_avg = assessment_performance.get('Assignment', avg_percentage)
                exam_avg = (assessment_performance.get('Midterm', avg_percentage) + 
                           assessment_performance.get('Final', avg_percentage)) / 2
                
                # Attendance metrics
                attendance_rate = student_attendance['is_present'].mean()
                recent_attendance = student_attendance.tail(30)['is_present'].mean()  # Last 30 days
                
                # Trend analysis (improvement/decline)
                recent_performance = student_performance.tail(10)['percentage'].mean()
                early_performance = student_performance.head(10)['percentage'].mean()
                performance_trend = recent_performance - early_performance
                
                # Risk indicators
                low_attendance = 1 if attendance_rate < 0.8 else 0
                failing_subjects = len(subject_performance[subject_performance < 50])
                consistent_poor_performance = 1 if std_percentage < 5 and avg_percentage < 60 else 0
                
                features = {
                    'student_id': student_id,
                    'grade_numeric': int(student['grade'].replace('th', '')),
                    'age': student['age'],
                    'gender_numeric': 1 if student['gender'] == 'Male' else 0,
                    'parent_education_numeric': self._encode_parent_education(student['parent_education']),
                    'family_income_numeric': {'Low': 0, 'Middle': 1, 'High': 2}[student['family_income']],
                    'internet_access': int(student['internet_access']),
                    'motivation_level': student['motivation_level'],
                    'study_hours_per_day': student['study_hours_per_day'],
                    'has_learning_difficulty': int(student['has_learning_difficulty']),
                    'extracurricular_activities': student['extracurricular_activities'],
                    
                    # Performance features
                    'avg_percentage': avg_percentage,
                    'std_percentage': std_percentage if not pd.isna(std_percentage) else 0,
                    'math_performance': math_performance,
                    'science_performance': science_performance,
                    'english_performance': english_performance,
                    'quiz_avg': quiz_avg,
                    'assignment_avg': assignment_avg,
                    'exam_avg': exam_avg,
                    
                    # Attendance features
                    'attendance_rate': attendance_rate,
                    'recent_attendance': recent_attendance,
                    
                    # Trend features
                    'performance_trend': performance_trend,
                    
                    # Risk indicators
                    'low_attendance': low_attendance,
                    'failing_subjects': failing_subjects,
                    'consistent_poor_performance': consistent_poor_performance,
                    
                    # Target variables
                    'at_risk': self._calculate_at_risk(avg_percentage, attendance_rate, failing_subjects),
                    'next_semester_performance': self._predict_next_performance(avg_percentage, performance_trend),
                    'needs_intervention': self._needs_intervention(avg_percentage, attendance_rate, failing_subjects, consistent_poor_performance)
                }
                
                student_features.append(features)
        
        self.feature_df = pd.DataFrame(student_features)
        print(f"Features prepared for {len(self.feature_df)} students")
        return self.feature_df
    
    def _encode_parent_education(self, education):
        """Encode parent education levels"""
        mapping = {
            'No Formal Education': 0,
            'High School': 1,
            'Bachelor': 2,
            'Master': 3,
            'PhD': 4
        }
        return mapping.get(education, 1)
    
    def _calculate_at_risk(self, avg_performance, attendance_rate, failing_subjects):
        """Calculate if student is at risk"""
        risk_score = 0
        
        if avg_performance < 60:
            risk_score += 2
        elif avg_performance < 70:
            risk_score += 1
            
        if attendance_rate < 0.75:
            risk_score += 2
        elif attendance_rate < 0.85:
            risk_score += 1
            
        if failing_subjects >= 2:
            risk_score += 2
        elif failing_subjects >= 1:
            risk_score += 1
            
        return 1 if risk_score >= 3 else 0
    
    def _predict_next_performance(self, current_performance, trend):
        """Predict next semester performance category"""
        predicted = current_performance + (trend * 0.5)  # Damped trend
        
        if predicted >= 85:
            return 'Excellent'
        elif predicted >= 75:
            return 'Good'
        elif predicted >= 60:
            return 'Average'
        else:
            return 'Below Average'
    
    def _needs_intervention(self, avg_performance, attendance_rate, failing_subjects, consistent_poor):
        """Determine if intervention is needed"""
        return 1 if (avg_performance < 65 or attendance_rate < 0.8 or 
                    failing_subjects >= 1 or consistent_poor) else 0
    
    def train_risk_prediction_model(self):
        """Train model to predict at-risk students"""
        print("Training at-risk prediction model...")
        
        # Prepare features
        feature_cols = [
            'grade_numeric', 'age', 'gender_numeric', 'parent_education_numeric',
            'family_income_numeric', 'internet_access', 'motivation_level',
            'study_hours_per_day', 'has_learning_difficulty', 'extracurricular_activities',
            'avg_percentage', 'std_percentage', 'math_performance', 'science_performance',
            'english_performance', 'attendance_rate', 'recent_attendance',
            'performance_trend', 'failing_subjects'
        ]
        
        X = self.feature_df[feature_cols]
        y = self.feature_df['at_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        self.models['risk_prediction'] = best_model
        self.scalers['risk_prediction'] = scaler if best_name == 'logistic_regression' else None
        
        print(f"Best model: {best_name} with accuracy: {best_score:.3f}")
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 5 most important features for risk prediction:")
            print(importance_df.head())
        
        return best_model, best_score
    
    def train_performance_prediction_model(self):
        """Train model to predict next semester performance"""
        print("\nTraining performance prediction model...")
        
        feature_cols = [
            'grade_numeric', 'parent_education_numeric', 'family_income_numeric',
            'motivation_level', 'study_hours_per_day', 'avg_percentage',
            'performance_trend', 'attendance_rate', 'failing_subjects'
        ]
        
        X = self.feature_df[feature_cols]
        y = self.feature_df['avg_percentage']  # Predict actual performance
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Performance prediction - MSE: {mse:.2f}, R²: {r2:.3f}")
        
        self.models['performance_prediction'] = model
        
        return model, r2
    
    def train_intervention_model(self):
        """Train model to determine intervention needs"""
        print("\nTraining intervention recommendation model...")
        
        feature_cols = [
            'avg_percentage', 'attendance_rate', 'failing_subjects',
            'consistent_poor_performance', 'has_learning_difficulty',
            'motivation_level', 'study_hours_per_day', 'performance_trend'
        ]
        
        X = self.feature_df[feature_cols]
        y = self.feature_df['needs_intervention']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Intervention model accuracy: {accuracy:.3f}")
        
        self.models['intervention'] = model
        
        return model, accuracy
    
    def predict_student_risk(self, student_id):
        """Predict if a specific student is at risk"""
        if 'risk_prediction' not in self.models:
            print("Risk prediction model not trained!")
            return None
        
        student_features = self.feature_df[self.feature_df['student_id'] == student_id]
        if len(student_features) == 0:
            print(f"Student {student_id} not found!")
            return None
        
        feature_cols = [
            'grade_numeric', 'age', 'gender_numeric', 'parent_education_numeric',
            'family_income_numeric', 'internet_access', 'motivation_level',
            'study_hours_per_day', 'has_learning_difficulty', 'extracurricular_activities',
            'avg_percentage', 'std_percentage', 'math_performance', 'science_performance',
            'english_performance', 'attendance_rate', 'recent_attendance',
            'performance_trend', 'failing_subjects'
        ]
        
        X = student_features[feature_cols]
        
        if self.scalers['risk_prediction']:
            X = self.scalers['risk_prediction'].transform(X)
        
        risk_prob = self.models['risk_prediction'].predict_proba(X)[0]
        risk_prediction = self.models['risk_prediction'].predict(X)[0]
        
        return {
            'student_id': student_id,
            'at_risk': bool(risk_prediction),
            'risk_probability': risk_prob[1],  # Probability of being at risk
            'confidence': max(risk_prob)
        }
    
    def get_intervention_recommendations(self, student_id):
        """Get personalized intervention recommendations"""
        student_features = self.feature_df[self.feature_df['student_id'] == student_id]
        if len(student_features) == 0:
            return []
        
        student = student_features.iloc[0]
        recommendations = []
        
        # Academic performance interventions
        if student['avg_percentage'] < 60:
            recommendations.append({
                'category': 'Academic Support',
                'recommendation': 'Enroll in tutoring program',
                'priority': 'High',
                'expected_impact': 'Significant improvement in grades'
            })
        
        if student['failing_subjects'] >= 2:
            recommendations.append({
                'category': 'Subject-Specific Help',
                'recommendation': 'One-on-one mentoring for struggling subjects',
                'priority': 'High',
                'expected_impact': 'Targeted improvement in weak areas'
            })
        
        # Attendance interventions
        if student['attendance_rate'] < 0.8:
            recommendations.append({
                'category': 'Attendance Support',
                'recommendation': 'Meet with counselor to address attendance issues',
                'priority': 'High',
                'expected_impact': 'Improved class participation and learning'
            })
        
        # Motivation and study habits
        if student['motivation_level'] < 0.6:
            recommendations.append({
                'category': 'Motivation',
                'recommendation': 'Goal-setting workshop and motivational counseling',
                'priority': 'Medium',
                'expected_impact': 'Increased engagement and effort'
            })
        
        if student['study_hours_per_day'] < 2:
            recommendations.append({
                'category': 'Study Skills',
                'recommendation': 'Study skills workshop and time management training',
                'priority': 'Medium',
                'expected_impact': 'Better learning efficiency'
            })
        
        # Learning difficulties support
        if student['has_learning_difficulty']:
            recommendations.append({
                'category': 'Special Support',
                'recommendation': 'Specialized learning support and accommodations',
                'priority': 'High',
                'expected_impact': 'Personalized learning approach'
            })
        
        # Family and socioeconomic support
        if student['family_income_numeric'] == 0:  # Low income
            recommendations.append({
                'category': 'Resources',
                'recommendation': 'Provide educational resources and technology access',
                'priority': 'Medium',
                'expected_impact': 'Equal learning opportunities'
            })
        
        return recommendations
    
    def save_models(self, model_dir='models'):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')
        
        for scaler_name, scaler in self.scalers.items():
            if scaler:
                joblib.dump(scaler, f'{model_dir}/{scaler_name}_scaler.pkl')
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir='models'):
        """Load saved models"""
        import os
        
        model_files = {
            'risk_prediction': f'{model_dir}/risk_prediction_model.pkl',
            'performance_prediction': f'{model_dir}/performance_prediction_model.pkl',
            'intervention': f'{model_dir}/intervention_model.pkl'
        }
        
        for model_name, filepath in model_files.items():
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
        
        # Load scalers
        scaler_file = f'{model_dir}/risk_prediction_scaler.pkl'
        if os.path.exists(scaler_file):
            self.scalers['risk_prediction'] = joblib.load(scaler_file)
        
        print("Models loaded successfully!")
    
    def get_analytics_summary(self):
        """Get comprehensive analytics summary"""
        if len(self.feature_df) == 0:
            return {}
        
        summary = {
            'total_students': len(self.feature_df),
            'at_risk_students': int(self.feature_df['at_risk'].sum()),
            'at_risk_percentage': round((self.feature_df['at_risk'].sum() / len(self.feature_df)) * 100, 1),
            'average_performance': round(self.feature_df['avg_percentage'].mean(), 1),
            'average_attendance': round(self.feature_df['attendance_rate'].mean() * 100, 1),
            'students_needing_intervention': int(self.feature_df['needs_intervention'].sum()),
            'top_performing_subjects': self._get_subject_performance_ranking(),
            'grade_wise_performance': self._get_grade_wise_stats(),
            'intervention_categories': self._get_intervention_stats()
        }
        
        return summary
    
    def _get_subject_performance_ranking(self):
        """Get subject performance ranking"""
        subjects = ['math_performance', 'science_performance', 'english_performance']
        subject_names = ['Mathematics', 'Science', 'English']
        
        rankings = []
        for i, subject in enumerate(subjects):
            avg_score = self.feature_df[subject].mean()
            rankings.append({'subject': subject_names[i], 'average_score': round(avg_score, 1)})
        
        return sorted(rankings, key=lambda x: x['average_score'], reverse=True)
    
    def _get_grade_wise_stats(self):
        """Get grade-wise statistics"""
        grade_stats = []
        for grade in sorted(self.feature_df['grade_numeric'].unique()):
            grade_data = self.feature_df[self.feature_df['grade_numeric'] == grade]
            stats = {
                'grade': f"{grade}th",
                'total_students': len(grade_data),
                'at_risk_count': int(grade_data['at_risk'].sum()),
                'average_performance': round(grade_data['avg_percentage'].mean(), 1),
                'average_attendance': round(grade_data['attendance_rate'].mean() * 100, 1)
            }
            grade_stats.append(stats)
        
        return grade_stats
    
    def _get_intervention_stats(self):
        """Get intervention statistics"""
        interventions_needed = self.feature_df[self.feature_df['needs_intervention'] == 1]
        
        categories = {
            'Academic Support': len(interventions_needed[interventions_needed['avg_percentage'] < 60]),
            'Attendance Issues': len(interventions_needed[interventions_needed['attendance_rate'] < 0.8]),
            'Learning Difficulties': len(interventions_needed[interventions_needed['has_learning_difficulty'] == 1]),
            'Low Motivation': len(interventions_needed[interventions_needed['motivation_level'] < 0.6])
        }
        
        return categories

def main():
    """Main function to train all models"""
    print("Starting EduInsights ML Training...")
    
    # Initialize ML system
    ml_system = EduAnalyticsML()
    
    # Load data
    ml_system.load_data()
    
    # Prepare features
    ml_system.prepare_features()
    
    # Train all models
    ml_system.train_risk_prediction_model()
    ml_system.train_performance_prediction_model()
    ml_system.train_intervention_model()
    
    # Save models
    ml_system.save_models()
    
    # Get analytics summary
    summary = ml_system.get_analytics_summary()
    
    print("\n" + "="*50)
    print("EDUINSIGHTS ANALYTICS SUMMARY")
    print("="*50)
    print(f"Total Students: {summary['total_students']}")
    print(f"At-Risk Students: {summary['at_risk_students']} ({summary['at_risk_percentage']}%)")
    print(f"Average Performance: {summary['average_performance']}%")
    print(f"Average Attendance: {summary['average_attendance']}%")
    print(f"Students Needing Intervention: {summary['students_needing_intervention']}")
    
    print("\nML models trained and saved successfully!")

if __name__ == "__main__":
    main() 