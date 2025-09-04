import unittest
import pandas as pd
import sqlite3
import os
import numpy as np
from data_generator import EduDataGenerator
from models import EduAnalyticsML
import tempfile

class TestEduInsights(unittest.TestCase):
    """Comprehensive test suite for EduInsights system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and data"""
        cls.test_db = 'test_eduinsights.db'
        cls.generator = EduDataGenerator(num_students=100, num_teachers=10, num_subjects=8)
        cls.students_df, cls.teachers_df, cls.performance_df, cls.attendance_df = cls.generator.create_database(cls.test_db)
        
        # Initialize ML system
        cls.ml_system = EduAnalyticsML(cls.test_db)
        cls.ml_system.load_data()
        cls.ml_system.prepare_features()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        
        # Clean up saved models
        if os.path.exists('saved_models'):
            import shutil
            shutil.rmtree('saved_models')
    
    def test_data_generation(self):
        """Test data generation functionality"""
        print("\n=== Testing Data Generation ===")
        
        # Test student data
        self.assertGreater(len(self.students_df), 0, "Students data should not be empty")
        self.assertIn('student_id', self.students_df.columns, "Student ID column should exist")
        self.assertIn('name', self.students_df.columns, "Name column should exist")
        
        # Test performance data
        self.assertGreater(len(self.performance_df), 0, "Performance data should not be empty")
        self.assertIn('student_id', self.performance_df.columns, "Student ID column should exist")
        self.assertIn('percentage', self.performance_df.columns, "Percentage column should exist")
        
        # Test attendance data
        self.assertGreater(len(self.attendance_df), 0, "Attendance data should not be empty")
        self.assertIn('student_id', self.attendance_df.columns, "Student ID column should exist")
        self.assertIn('is_present', self.attendance_df.columns, "Is present column should exist")
        
        # Test teachers data
        self.assertGreater(len(self.teachers_df), 0, "Teachers data should not be empty")
        self.assertIn('teacher_id', self.teachers_df.columns, "Teacher ID column should exist")
        
        print("✓ Data generation tests passed")
    
    def test_data_quality(self):
        """Test data quality and consistency"""
        print("\n=== Testing Data Quality ===")
        
        # Test performance scores are within valid range
        self.assertTrue(
            self.performance_df['percentage'].between(0, 100).all(),
            "All performance scores should be between 0 and 100"
        )
        
        # Test attendance data is boolean
        self.assertTrue(
            self.attendance_df['is_present'].isin([True, False]).all(),
            "Attendance data should be boolean"
        )
        
        # Test student IDs are consistent across tables
        student_ids_students = set(self.students_df['student_id'])
        student_ids_performance = set(self.performance_df['student_id'])
        student_ids_attendance = set(self.attendance_df['student_id'])
        
        self.assertTrue(
            student_ids_performance.issubset(student_ids_students),
            "Performance data should only contain valid student IDs"
        )
        self.assertTrue(
            student_ids_attendance.issubset(student_ids_students),
            "Attendance data should only contain valid student IDs"
        )
        
        # Test for reasonable data distributions
        avg_performance = self.performance_df['percentage'].mean()
        self.assertGreater(avg_performance, 30, "Average performance should be reasonable")
        self.assertLess(avg_performance, 95, "Average performance should be realistic")
        
        print("✓ Data quality tests passed")
    
    def test_ml_feature_preparation(self):
        """Test ML feature preparation"""
        print("\n=== Testing ML Feature Preparation ===")
        
        features_df = self.ml_system.features_df
        
        # Test features dataframe exists and has data
        self.assertIsNotNone(features_df, "Features dataframe should exist")
        self.assertGreater(len(features_df), 0, "Features dataframe should not be empty")
        
        # Test required columns exist
        required_columns = [
            'student_id', 'grade', 'avg_performance', 'attendance_rate',
            'at_risk', 'needs_intervention'
        ]
        
        for col in required_columns:
            self.assertIn(col, features_df.columns, f"Column {col} should exist in features")
        
        # Test data types and ranges
        self.assertTrue(
            features_df['avg_performance'].between(0, 100).all(),
            "Average performance should be between 0 and 100"
        )
        
        self.assertTrue(
            features_df['attendance_rate'].between(0, 1).all(),
            "Attendance rate should be between 0 and 1"
        )
        
        self.assertTrue(
            features_df['at_risk'].isin([0, 1]).all(),
            "At-risk column should be binary"
        )
        
        print("✓ ML feature preparation tests passed")
    
    def test_risk_prediction_model(self):
        """Test risk prediction model"""
        print("\n=== Testing Risk Prediction Model ===")
        
        # Train model
        model = self.ml_system.train_risk_model()
        
        # Test model exists
        self.assertIsNotNone(model, "Risk prediction model should be created")
        self.assertIn('risk', self.ml_system.models, "Risk model should be stored")
        
        # Test model predictions
        sample_student = self.ml_system.features_df.iloc[0]
        risk_prob = self.ml_system.predict_risk(sample_student['student_id'])
        
        self.assertIsNotNone(risk_prob, "Risk prediction should return a value")
        self.assertGreaterEqual(risk_prob, 0, "Risk probability should be >= 0")
        self.assertLessEqual(risk_prob, 1, "Risk probability should be <= 1")
        
        print("✓ Risk prediction model tests passed")
    
    def test_performance_prediction_model(self):
        """Test performance prediction model"""
        print("\n=== Testing Performance Prediction Model ===")
        
        # Train model
        model = self.ml_system.train_performance_model()
        
        # Test model exists
        self.assertIsNotNone(model, "Performance prediction model should be created")
        self.assertIn('performance', self.ml_system.models, "Performance model should be stored")
        
        # Test model can make predictions
        sample_features = self.ml_system.features_df[['grade', 'motivation', 'study_hours', 'attendance_rate']].iloc[0:1]
        prediction = model.predict(sample_features)
        
        self.assertIsNotNone(prediction, "Performance prediction should return a value")
        self.assertGreater(prediction[0], 0, "Performance prediction should be positive")
        self.assertLess(prediction[0], 100, "Performance prediction should be realistic")
        
        print("✓ Performance prediction model tests passed")
    
    def test_intervention_recommendations(self):
        """Test intervention recommendation system"""
        print("\n=== Testing Intervention Recommendations ===")
        
        # Test recommendations for different student types
        for i in range(min(5, len(self.ml_system.features_df))):
            student_id = self.ml_system.features_df.iloc[i]['student_id']
            recommendations = self.ml_system.get_intervention_recommendations(student_id)
            
            self.assertIsInstance(recommendations, list, "Recommendations should be a list")
            self.assertGreater(len(recommendations), 0, "Should have at least one recommendation")
            
            # Test that recommendations are strings
            for rec in recommendations:
                self.assertIsInstance(rec, str, "Each recommendation should be a string")
                self.assertGreater(len(rec), 0, "Recommendations should not be empty")
        
        print("✓ Intervention recommendation tests passed")
    
    def test_analytics_summary(self):
        """Test analytics summary generation"""
        print("\n=== Testing Analytics Summary ===")
        
        summary = self.ml_system.get_analytics_summary()
        
        # Test summary structure
        required_keys = [
            'total_students', 'at_risk_count', 'at_risk_percentage',
            'avg_performance', 'avg_attendance', 'intervention_needed'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary, f"Summary should contain {key}")
        
        # Test summary values
        self.assertGreater(summary['total_students'], 0, "Total students should be positive")
        self.assertGreaterEqual(summary['at_risk_count'], 0, "At-risk count should be non-negative")
        self.assertGreaterEqual(summary['at_risk_percentage'], 0, "At-risk percentage should be non-negative")
        self.assertLessEqual(summary['at_risk_percentage'], 100, "At-risk percentage should be <= 100")
        
        print("✓ Analytics summary tests passed")
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        print("\n=== Testing Model Persistence ===")
        
        # Train models
        self.ml_system.train_risk_model()
        self.ml_system.train_performance_model()
        
        # Save models
        self.ml_system.save_models()
        
        # Test model files exist
        self.assertTrue(os.path.exists('saved_models'), "Saved models directory should exist")
        self.assertTrue(os.path.exists('saved_models/risk_model.pkl'), "Risk model file should exist")
        self.assertTrue(os.path.exists('saved_models/performance_model.pkl'), "Performance model file should exist")
        
        # Test loading models
        new_ml_system = EduAnalyticsML(self.test_db)
        new_ml_system.load_data()
        new_ml_system.prepare_features()
        new_ml_system.load_models()
        
        self.assertIn('risk', new_ml_system.models, "Risk model should be loaded")
        self.assertIn('performance', new_ml_system.models, "Performance model should be loaded")
        
        print("✓ Model persistence tests passed")
    
    def test_dashboard_data_loading(self):
        """Test dashboard data loading functions"""
        print("\n=== Testing Dashboard Data Loading ===")
        
        # Test database connection
        conn = sqlite3.connect(self.test_db)
        
        # Test loading each table
        students_df = pd.read_sql_query("SELECT * FROM students", conn)
        performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
        attendance_df = pd.read_sql_query("SELECT * FROM attendance", conn)
        teachers_df = pd.read_sql_query("SELECT * FROM teachers", conn)
        
        conn.close()
        
        # Verify data loaded correctly
        self.assertGreater(len(students_df), 0, "Students data should load")
        self.assertGreater(len(performance_df), 0, "Performance data should load")
        self.assertGreater(len(attendance_df), 0, "Attendance data should load")
        self.assertGreater(len(teachers_df), 0, "Teachers data should load")
        
        print("✓ Dashboard data loading tests passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n=== Testing Edge Cases ===")
        
        # Test with non-existent student ID
        invalid_recommendations = self.ml_system.get_intervention_recommendations('INVALID_ID')
        self.assertEqual(len(invalid_recommendations), 0, "Should return empty list for invalid student")
        
        # Test risk prediction with invalid student
        invalid_risk = self.ml_system.predict_risk('INVALID_ID')
        self.assertIsNone(invalid_risk, "Should return None for invalid student")
        
        # Test with empty data
        empty_ml = EduAnalyticsML('nonexistent.db')
        try:
            empty_ml.load_data()
            # Should handle gracefully
        except Exception as e:
            # Expected to fail, but shouldn't crash
            pass
        
        print("✓ Edge case tests passed")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n=== Testing Performance Benchmarks ===")
        
        import time
        
        # Test data generation speed
        start_time = time.time()
        test_generator = EduDataGenerator(num_students=50, num_teachers=5, num_subjects=4)
        test_generator.generate_students_data()
        generation_time = time.time() - start_time
        self.assertLess(generation_time, 5, "Data generation should complete within 5 seconds")
        
        # Test ML training speed
        start_time = time.time()
        self.ml_system.train_risk_model()
        training_time = time.time() - start_time
        self.assertLess(training_time, 10, "Model training should complete within 10 seconds")
        
        # Test prediction speed
        start_time = time.time()
        for i in range(10):
            student_id = self.ml_system.features_df.iloc[i % len(self.ml_system.features_df)]['student_id']
            self.ml_system.predict_risk(student_id)
        prediction_time = time.time() - start_time
        self.assertLess(prediction_time, 1, "10 predictions should complete within 1 second")
        
        print("✓ Performance benchmark tests passed")

def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("="*60)
    print("🎓 EDUINSIGHTS COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEduInsights)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {successes}")
    print(f"❌ Failed: {failures}")
    print(f"🔥 Errors: {errors}")
    print(f"Success Rate: {(successes/total_tests)*100:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("\n" + "="*60)
    if failures == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1) 