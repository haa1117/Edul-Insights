#!/usr/bin/env python3
"""
EduInsights - Academic Analytics Dashboard
Main application runner
"""

import os
import sys
import subprocess
from data_generator import EduDataGenerator
from models import train_all_models
import sqlite3

def check_database_exists():
    """Check if database exists, create if not"""
    if not os.path.exists('eduinsights.db'):
        print("🔧 Database not found. Creating new database with sample data...")
        generator = EduDataGenerator(num_students=500, num_teachers=25, num_subjects=8)
        generator.create_database()
        print("✅ Database created successfully!")
        return True
    else:
        print("✅ Database found!")
        return False

def check_models_exist():
    """Check if ML models exist, train if not"""
    if not os.path.exists('saved_models') or not os.listdir('saved_models'):
        print("🤖 ML models not found. Training new models...")
        train_all_models()
        print("✅ ML models trained successfully!")
        return True
    else:
        print("✅ ML models found!")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    return True

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    try:
        from tests import run_tests
        success = run_tests()
        if success:
            print("✅ All tests passed!")
        else:
            print("⚠️  Some tests failed, but system should still work")
        return success
    except ImportError:
        print("⚠️  Test module not found, skipping tests")
        return True

def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("🎓 WELCOME TO EDUINSIGHTS")
    print("   Academic Analytics Dashboard")
    print("="*60)
    print()
    print("Features:")
    print("• 📊 Real-time student performance monitoring")
    print("• ⚠️  At-risk student identification")
    print("• 🎯 Predictive analytics and interventions")
    print("• 📈 Comprehensive reporting and analytics")
    print("• 👥 Student and teacher management")
    print("• 📅 Attendance tracking and analysis")
    print()

def get_system_info():
    """Get and display system information"""
    try:
        conn = sqlite3.connect('eduinsights.db')
        
        # Get database stats
        students_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        performance_records = conn.execute("SELECT COUNT(*) FROM performance").fetchone()[0]
        attendance_records = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        teachers_count = conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
        
        conn.close()
        
        print("📊 System Statistics:")
        print(f"   Students: {students_count:,}")
        print(f"   Teachers: {teachers_count:,}")
        print(f"   Performance Records: {performance_records:,}")
        print(f"   Attendance Records: {attendance_records:,}")
        print()
        
    except Exception as e:
        print(f"⚠️  Could not retrieve system stats: {e}")

def main():
    """Main application runner"""
    print_welcome()
    
    # Setup system
    print("🚀 Setting up EduInsights...")
    
    # Check and install dependencies
    if not install_dependencies():
        return False
    
    # Check database
    db_created = check_database_exists()
    
    # Check models
    models_trained = check_models_exist()
    
    # Run tests if new setup
    if db_created or models_trained:
        run_tests()
    
    # Display system info
    get_system_info()
    
    print("✅ EduInsights setup complete!")
    print()
    print("🌐 Starting dashboard server...")
    print("   Dashboard will be available at: http://localhost:8050")
    print("   Press Ctrl+C to stop the server")
    print()
    
    # Import and run dashboard
    try:
        from dashboard import app
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n👋 EduInsights dashboard stopped. Thank you for using our system!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("Please check the error and try again.")
        return False
    
    return True

def setup_only():
    """Setup system without running dashboard"""
    print_welcome()
    print("🔧 Setting up EduInsights (setup only)...")
    
    install_dependencies()
    check_database_exists()
    check_models_exist()
    run_tests()
    get_system_info()
    
    print("✅ EduInsights setup complete!")
    print("Run 'python app.py' to start the dashboard")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_only()
    else:
        main() 