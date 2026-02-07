#!/usr/bin/env python3
"""
Test script for Legal Document Retrieval System
Run this to verify all components are working correctly
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path

def test_file_existence():
    """Test if all required files exist"""
    print("🔍 Testing file existence...")
    
    required_files = {
        "train_faiss_index_legalbert.idx": "FAISS Index",
        "train_document_map_legalbert.npy": "Document Map", 
        "legal_ner_output.json": "NER Data",
        "cleaned_texts": "Cleaned Texts Folder",
        "nlp ds": "PDF Folder"
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {description}: Found")
        else:
            print(f"  ❌ {description}: Missing ({file_path})")
            all_exist = False
    
    return all_exist

def test_faiss_index():
    """Test FAISS index loading"""
    print("\n🔍 Testing FAISS index...")
    
    try:
        index = faiss.read_index("train_faiss_index_legalbert.idx")
        print(f"  ✅ FAISS index loaded successfully")
        print(f"  📊 Total documents: {index.ntotal}")
        print(f"  📏 Vector dimension: {index.d}")
        return True
    except Exception as e:
        print(f"  ❌ Error loading FAISS index: {e}")
        return False

def test_document_map():
    """Test document map loading"""
    print("\n🔍 Testing document map...")
    
    try:
        doc_map = np.load("train_document_map_legalbert.npy", allow_pickle=True)
        print(f"  ✅ Document map loaded successfully")
        print(f"  📊 Total entries: {len(doc_map)}")
        print(f"  📄 Sample documents: {list(doc_map[:3])}")
        return True
    except Exception as e:
        print(f"  ❌ Error loading document map: {e}")
        return False

def test_ner_data():
    """Test NER data loading"""
    print("\n🔍 Testing NER data...")
    
    try:
        with open("legal_ner_output.json", 'r', encoding='utf-8') as f:
            ner_data = json.load(f)
        
        print(f"  ✅ NER data loaded successfully")
        print(f"  📊 Total entities: {len(ner_data)}")
        
        # Count unique files
        unique_files = set(entity['file'] for entity in ner_data)
        print(f"  📄 Files with entities: {len(unique_files)}")
        
        # Count entity types
        entity_types = {}
        for entity in ner_data[:1000]:  # Sample first 1000
            label = entity['entity_label']
            entity_types[label] = entity_types.get(label, 0) + 1
        
        print(f"  🏷️  Entity types found: {list(entity_types.keys())[:5]}...")
        return True
    except Exception as e:
        print(f"  ❌ Error loading NER data: {e}")
        return False

def test_cleaned_texts():
    """Test cleaned texts folder"""
    print("\n🔍 Testing cleaned texts...")
    
    try:
        cleaned_folder = Path("cleaned_texts")
        if not cleaned_folder.exists():
            print(f"  ❌ Cleaned texts folder not found")
            return False
        
        txt_files = list(cleaned_folder.glob("*.txt"))
        print(f"  ✅ Cleaned texts folder found")
        print(f"  📊 Total text files: {len(txt_files)}")
        
        # Test reading a sample file
        if txt_files:
            sample_file = txt_files[0]
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"  📄 Sample file: {sample_file.name} ({len(content)} characters)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error testing cleaned texts: {e}")
        return False

def test_pdf_folder():
    """Test PDF folder structure"""
    print("\n🔍 Testing PDF folder...")
    
    try:
        pdf_folder = Path("nlp ds")
        if not pdf_folder.exists():
            print(f"  ❌ PDF folder not found")
            return False
        
        print(f"  ✅ PDF folder found")
        
        # Check subfolders
        subfolders = [d for d in pdf_folder.iterdir() if d.is_dir()]
        print(f"  📁 Subfolders: {[d.name for d in subfolders]}")
        
        # Count PDF files
        total_pdfs = 0
        for subfolder in subfolders:
            pdf_files = list(subfolder.glob("*.pdf"))
            total_pdfs += len(pdf_files)
            print(f"    📄 {subfolder.name}: {len(pdf_files)} PDFs")
        
        print(f"  📊 Total PDF files: {total_pdfs}")
        return True
    except Exception as e:
        print(f"  ❌ Error testing PDF folder: {e}")
        return False

def test_dependencies():
    """Test Python dependencies"""
    print("\n🔍 Testing Python dependencies...")
    
    dependencies = [
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
        ("faiss", "FAISS"),
        ("sentence_transformers", "Sentence Transformers"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch")
    ]
    
    all_available = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}: Available")
        except ImportError:
            print(f"  ❌ {display_name}: Missing ({module_name})")
            all_available = False
    
    return all_available

def main():
    """Run all tests"""
    print("⚖️  Legal Document Retrieval System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("File Existence", test_file_existence),
        ("FAISS Index", test_faiss_index),
        ("Document Map", test_document_map),
        ("NER Data", test_ner_data),
        ("Cleaned Texts", test_cleaned_texts),
        ("PDF Folder", test_pdf_folder),
        ("Dependencies", test_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("🚀 Run 'python run_app.py' to start the application.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("📖 Refer to README.md for troubleshooting guidance.")

if __name__ == "__main__":
    main()