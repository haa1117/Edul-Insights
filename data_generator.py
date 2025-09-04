import pandas as pd
import numpy as np
import sqlite3
from faker import Faker
import random
from datetime import datetime, timedelta
import json

class EduDataGenerator:
    def __init__(self, num_students=500, num_teachers=25, num_subjects=8):
        self.fake = Faker()
        self.num_students = num_students
        self.num_teachers = num_teachers
        self.num_subjects = num_subjects
        
        # Educational parameters
        self.subjects = [
            'Mathematics', 'Science', 'English', 'History', 
            'Geography', 'Physics', 'Chemistry', 'Biology'
        ]
        self.grades = ['9th', '10th', '11th', '12th']
        self.sections = ['A', 'B', 'C', 'D']
        
        # Academic calendar
        self.academic_start = datetime(2023, 8, 1)
        self.academic_end = datetime(2024, 6, 30)
        
    def generate_students_data(self):
        """Generate comprehensive student profiles"""
        students = []
        
        for i in range(self.num_students):
            # Basic demographics with socioeconomic factors
            gender = random.choice(['Male', 'Female'])
            age = random.randint(14, 18)
            grade = random.choice(self.grades)
            section = random.choice(self.sections)
            
            # Socioeconomic and background factors that affect performance
            parent_education = random.choice([
                'High School', 'Bachelor', 'Master', 'PhD', 'No Formal Education'
            ])
            family_income = random.choice(['Low', 'Middle', 'High'])
            internet_access = random.choice([True, False]) if family_income == 'Low' else True
            
            # Learning characteristics
            learning_style = random.choice(['Visual', 'Auditory', 'Kinesthetic', 'Mixed'])
            motivation_level = random.uniform(0.3, 1.0)
            study_hours_per_day = max(0.5, np.random.normal(3, 1.5))
            
            # Health and personal factors
            has_learning_difficulty = random.choice([True, False]) if random.random() < 0.15 else False
            extracurricular_activities = random.randint(0, 4)
            
            student = {
                'student_id': f'STU{i+1:04d}',
                'name': self.fake.name(),
                'gender': gender,
                'age': age,
                'grade': grade,
                'section': section,
                'parent_education': parent_education,
                'family_income': family_income,
                'internet_access': internet_access,
                'learning_style': learning_style,
                'motivation_level': round(motivation_level, 2),
                'study_hours_per_day': round(study_hours_per_day, 1),
                'has_learning_difficulty': has_learning_difficulty,
                'extracurricular_activities': extracurricular_activities,
                'enrollment_date': self.fake.date_between(
                    start_date=self.academic_start, 
                    end_date=self.academic_start + timedelta(days=30)
                )
            }
            students.append(student)
            
        return pd.DataFrame(students)
    
    def generate_teachers_data(self):
        """Generate teacher profiles"""
        teachers = []
        
        for i in range(self.num_teachers):
            teacher = {
                'teacher_id': f'TCH{i+1:03d}',
                'name': self.fake.name(),
                'subject': random.choice(self.subjects),
                'experience_years': random.randint(1, 25),
                'qualification': random.choice(['Bachelor', 'Master', 'PhD']),
                'teaching_load': random.randint(15, 30),  # hours per week
                'performance_rating': round(random.uniform(3.0, 5.0), 1)
            }
            teachers.append(teacher)
            
        return pd.DataFrame(teachers)
    
    def generate_performance_data(self, students_df, teachers_df):
        """Generate realistic academic performance data"""
        performances = []
        
        # Create semester structure
        semesters = ['Semester 1', 'Semester 2']
        months = {
            'Semester 1': [8, 9, 10, 11, 12],  # Aug to Dec
            'Semester 2': [1, 2, 3, 4, 5, 6]   # Jan to June
        }
        
        for _, student in students_df.iterrows():
            # Student's base ability (affects all subjects)
            base_ability = self._calculate_base_ability(student)
            
            for semester in semesters:
                for subject in self.subjects:
                    # Get teacher for this subject
                    subject_teachers = teachers_df[teachers_df['subject'] == subject]
                    if len(subject_teachers) > 0:
                        teacher = subject_teachers.sample(1).iloc[0]
                    else:
                        # Assign a random teacher if no subject-specific teacher
                        teacher = teachers_df.sample(1).iloc[0]
                    
                    # Generate multiple assessments per semester
                    for month in months[semester]:
                        # Quiz scores (2 per month)
                        for quiz_num in range(2):
                            score = self._generate_realistic_score(
                                student, subject, teacher, 'Quiz', base_ability
                            )
                            performances.append({
                                'student_id': student['student_id'],
                                'teacher_id': teacher['teacher_id'],
                                'subject': subject,
                                'semester': semester,
                                'assessment_type': 'Quiz',
                                'assessment_date': datetime(2023 if month >= 8 else 2024, month, random.randint(1, 28)),
                                'max_score': 20,
                                'obtained_score': score,
                                'percentage': round((score/20)*100, 1)
                            })
                        
                        # Assignment (1 per month)
                        score = self._generate_realistic_score(
                            student, subject, teacher, 'Assignment', base_ability
                        )
                        performances.append({
                            'student_id': student['student_id'],
                            'teacher_id': teacher['teacher_id'],
                            'subject': subject,
                            'semester': semester,
                            'assessment_type': 'Assignment',
                            'assessment_date': datetime(2023 if month >= 8 else 2024, month, random.randint(1, 28)),
                            'max_score': 50,
                            'obtained_score': score,
                            'percentage': round((score/50)*100, 1)
                        })
                
                # Major exams per semester
                for exam_type in ['Midterm', 'Final']:
                    for subject in self.subjects:
                        subject_teachers = teachers_df[teachers_df['subject'] == subject]
                        teacher = subject_teachers.sample(1).iloc[0] if len(subject_teachers) > 0 else teachers_df.sample(1).iloc[0]
                        
                        score = self._generate_realistic_score(
                            student, subject, teacher, exam_type, base_ability
                        )
                        exam_month = 12 if exam_type == 'Midterm' and semester == 'Semester 1' else \
                                    3 if exam_type == 'Midterm' and semester == 'Semester 2' else \
                                    1 if semester == 'Semester 1' else 6
                        
                        performances.append({
                            'student_id': student['student_id'],
                            'teacher_id': teacher['teacher_id'],
                            'subject': subject,
                            'semester': semester,
                            'assessment_type': exam_type,
                            'assessment_date': datetime(2023 if exam_month >= 8 else 2024, exam_month, random.randint(1, 28)),
                            'max_score': 100,
                            'obtained_score': score,
                            'percentage': round((score/100)*100, 1)
                        })
        
        return pd.DataFrame(performances)
    
    def generate_attendance_data(self, students_df):
        """Generate realistic attendance data"""
        attendance_records = []
        
        # Generate daily attendance for academic year
        current_date = self.academic_start
        while current_date <= self.academic_end:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                for _, student in students_df.iterrows():
                    # Base attendance probability based on student characteristics
                    base_attendance_prob = self._calculate_attendance_probability(student)
                    
                    # Seasonal variations (lower attendance in winter/exams)
                    seasonal_factor = 1.0
                    if current_date.month in [12, 1]:  # Winter break effect
                        seasonal_factor = 0.85
                    elif current_date.month in [3, 6]:  # Exam periods
                        seasonal_factor = 0.95
                    
                    final_prob = base_attendance_prob * seasonal_factor
                    is_present = random.random() < final_prob
                    
                    attendance_records.append({
                        'student_id': student['student_id'],
                        'date': current_date,
                        'is_present': is_present,
                        'reason_for_absence': self._get_absence_reason() if not is_present else None
                    })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(attendance_records)
    
    def _calculate_base_ability(self, student):
        """Calculate student's base academic ability based on background factors"""
        ability = 0.5  # Base ability
        
        # Family income effect
        if student['family_income'] == 'High':
            ability += 0.2
        elif student['family_income'] == 'Low':
            ability -= 0.15
        
        # Parent education effect
        education_bonus = {
            'PhD': 0.25, 'Master': 0.2, 'Bachelor': 0.15, 
            'High School': 0.05, 'No Formal Education': -0.1
        }
        ability += education_bonus.get(student['parent_education'], 0)
        
        # Other factors
        ability += student['motivation_level'] * 0.3
        ability += min(student['study_hours_per_day'] / 5, 0.2)  # Cap at 0.2
        
        if not student['internet_access']:
            ability -= 0.1
        if student['has_learning_difficulty']:
            ability -= 0.15
        
        return max(0.1, min(1.0, ability))  # Keep between 0.1 and 1.0
    
    def _generate_realistic_score(self, student, subject, teacher, assessment_type, base_ability):
        """Generate realistic scores based on multiple factors"""
        # Subject difficulty multipliers
        subject_difficulty = {
            'Mathematics': 0.85, 'Physics': 0.82, 'Chemistry': 0.83,
            'Biology': 0.88, 'Science': 0.87, 'English': 0.90,
            'History': 0.92, 'Geography': 0.91
        }
        
        # Assessment type difficulty
        assessment_difficulty = {
            'Quiz': 0.95, 'Assignment': 0.90, 'Midterm': 0.85, 'Final': 0.80
        }
        
        # Teacher effect
        teacher_effect = (teacher['performance_rating'] - 3.0) / 10  # -0.2 to +0.2
        
        # Calculate expected score
        difficulty_factor = subject_difficulty.get(subject, 0.85) * assessment_difficulty.get(assessment_type, 0.85)
        expected_performance = (base_ability + teacher_effect) * difficulty_factor
        
        # Add some randomness
        randomness = np.random.normal(0, 0.1)
        final_performance = max(0.1, min(1.0, expected_performance + randomness))
        
        # Convert to score based on assessment type
        max_scores = {'Quiz': 20, 'Assignment': 50, 'Midterm': 100, 'Final': 100}
        max_score = max_scores.get(assessment_type, 100)
        
        return round(final_performance * max_score)
    
    def _calculate_attendance_probability(self, student):
        """Calculate attendance probability based on student factors"""
        prob = 0.85  # Base attendance rate
        
        # Motivation effect
        prob += student['motivation_level'] * 0.1
        
        # Family income effect (transportation, health)
        if student['family_income'] == 'High':
            prob += 0.05
        elif student['family_income'] == 'Low':
            prob -= 0.1
        
        # Learning difficulty effect
        if student['has_learning_difficulty']:
            prob -= 0.05
        
        return max(0.3, min(0.98, prob))  # Keep reasonable bounds
    
    def _get_absence_reason(self):
        """Generate realistic absence reasons"""
        reasons = [
            'Illness', 'Family Emergency', 'Medical Appointment', 
            'Transportation Issues', 'Weather', 'Personal Reasons'
        ]
        return random.choice(reasons)
    
    def create_database(self, db_name='eduinsights.db'):
        """Create SQLite database and populate with generated data"""
        print("Generating educational data...")
        
        # Generate all data
        students_df = self.generate_students_data()
        teachers_df = self.generate_teachers_data()
        performance_df = self.generate_performance_data(students_df, teachers_df)
        attendance_df = self.generate_attendance_data(students_df)
        
        print(f"Generated data for {len(students_df)} students, {len(teachers_df)} teachers")
        print(f"Created {len(performance_df)} performance records and {len(attendance_df)} attendance records")
        
        # Create database
        conn = sqlite3.connect(db_name)
        
        # Save to database
        students_df.to_sql('students', conn, if_exists='replace', index=False)
        teachers_df.to_sql('teachers', conn, if_exists='replace', index=False)
        performance_df.to_sql('performance', conn, if_exists='replace', index=False)
        attendance_df.to_sql('attendance', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Database '{db_name}' created successfully!")
        
        return students_df, teachers_df, performance_df, attendance_df

if __name__ == "__main__":
    # Generate data
    generator = EduDataGenerator(num_students=500, num_teachers=25, num_subjects=8)
    generator.create_database()
    print("Data generation completed!") 