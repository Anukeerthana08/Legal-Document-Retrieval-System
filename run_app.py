#!/usr/bin/env python3
"""
Legal Document Retrieval System Launcher
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "train_faiss_index_legalbert.idx",
        "train_document_map_legalbert.npy", 
        "legal_ner_output.json",
        "cleaned_texts",
        "nlp ds"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files/folders:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are in the current directory.")
        return False
    
    print("✅ All required files found!")
    return True

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting Legal Document Retrieval System...")
        print("📱 The application will open in your default web browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "simple_legal_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    print("⚖️  Legal Document Retrieval System")
    print("=" * 50)
    
    # Check if required files exist
    if not check_requirements():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()