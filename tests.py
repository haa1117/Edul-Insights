import unittest
import pandas as pd
import sqlite3
import os
from data_generator import EduDataGenerator
from models import EduAnalyticsML

class TestEduInsights(unittest.TestCase):
    """Test suite for EduInsights system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_db = 'test_eduinsights.db'
        cls.generator = EduDataGenerator(num_students=50, num_teachers=8, num_subjects=6)
        cls.students_df, cls.teachers_df, cls.performance_df, cls.attendance_df = cls.generator.create_database(cls.test_db)
        
        cls.ml_system = EduAnalyticsML(cls.test_db)
        cls.ml_system.load_data()
        cls.ml_system.prepare_features()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
    
    def test_data_generation(self):
        """Test data generation"""
        self.assertGreater(len(self.students_df), 0)
        self.assertGreater(len(self.performance_df), 0)
        self.assertGreater(len(self.attendance_df), 0)
        self.assertIn('student_id', self.students_df.columns)
        print("✓ Data generation test passed")
    
    def test_ml_models(self):
        """Test ML model training"""
        risk_model = self.ml_system.train_risk_model()
        perf_model = self.ml_system.train_performance_model()
        
        self.assertIsNotNone(risk_model)
        self.assertIsNotNone(perf_model)
        print("✓ ML models test passed")
    
    def test_predictions(self):
        """Test prediction functionality"""
        student_id = self.ml_system.features_df.iloc[0]['student_id']
        
        # Test risk prediction
        risk_prob = self.ml_system.predict_risk(student_id)
        self.assertIsNotNone(risk_prob)
        self.assertGreaterEqual(risk_prob, 0)
        self.assertLessEqual(risk_prob, 1)
        
        # Test interventions
        recommendations = self.ml_system.get_intervention_recommendations(student_id)
        self.assertIsInstance(recommendations, list)
        
        print("✓ Predictions test passed")
    
    def test_analytics(self):
        """Test analytics summary"""
        summary = self.ml_system.get_analytics_summary()
        
        required_keys = ['total_students', 'at_risk_count', 'avg_performance']
        for key in required_keys:
            self.assertIn(key, summary)
        
        print("✓ Analytics test passed")

def run_tests():
    """Run all tests"""
    print("🎓 Running EduInsights Tests...")
    print("="*50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEduInsights)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests() 