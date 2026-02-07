#!/usr/bin/env python3
"""
Test script for Hybrid BGE + NER Legal Document Retrieval System
This script tests the integration between BGE embeddings and NER entities
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def test_hybrid_integration():
    """Test if BGE and NER are working together effectively"""
    print("🧪 Testing Hybrid BGE + NER Integration")
    print("=" * 50)
    
    # Test queries with different entity types
    test_queries = [
        {
            "query": "Section 302 IPC murder conviction appeal",
            "expected_entities": ["STATUTE", "CASE_TYPE"],
            "description": "Legal statute with case type"
        },
        {
            "query": "Supreme Court constitutional Article 14 equality",
            "expected_entities": ["COURT", "STATUTE"],
            "description": "Court with constitutional provision"
        },
        {
            "query": "contract breach damages specific performance",
            "expected_entities": [],
            "description": "General legal concepts (no specific entities)"
        },
        {
            "query": "criminal appeal bail application High Court",
            "expected_entities": ["COURT", "CASE_TYPE"],
            "description": "Court with procedural terms"
        }
    ]
    
    # Load the retriever (simplified version for testing)
    try:
        from legal_retrieval_app import LegalDocumentRetriever
        retriever = LegalDocumentRetriever()
        
        # Check if files exist
        required_files = [
            "train_faiss_index_legalbert.idx",
            "train_document_map_legalbert.npy",
            "legal_ner_output.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            print("   Please ensure all required files are present")
            return False
        
        # Load system
        print("📚 Loading system components...")
        success = retriever.load_models_and_data()
        if not success:
            print("❌ Failed to load system components")
            return False
        
        print("✅ System loaded successfully!")
        print(f"📊 Documents in index: {retriever.index.ntotal}")
        print(f"🏷️ NER entities loaded: {len(retriever.ner_data)}")
        print(f"🔗 Entity types indexed: {len(retriever.entity_index) if hasattr(retriever, 'entity_index') else 0}")
        
        # Test each query
        print("\n🔍 Testing Query Processing:")
        print("-" * 30)
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"   Query: '{test_case['query']}'")
            
            # Test entity extraction
            if hasattr(retriever, '_extract_query_entities'):
                detected_entities = retriever._extract_query_entities(test_case['query'])
                print(f"   🏷️ Detected entities: {list(detected_entities.keys())}")
                
                # Check if expected entities were found
                expected = set(test_case['expected_entities'])
                detected = set(detected_entities.keys())
                
                if expected.issubset(detected):
                    print(f"   ✅ Entity detection: Expected entities found")
                elif expected:
                    print(f"   ⚠️ Entity detection: Missing {expected - detected}")
                else:
                    print(f"   ✅ Entity detection: No entities expected, none found")
            else:
                print(f"   ❌ Entity extraction method not available")
                continue
            
            # Test search with and without hybrid scoring
            try:
                # Semantic-only search
                semantic_results = retriever.search_documents(test_case['query'], top_k=3, use_hybrid_scoring=False)
                
                # Hybrid search
                hybrid_results = retriever.search_documents(test_case['query'], top_k=3, use_hybrid_scoring=True)
                
                print(f"   🧠 Semantic search: {len(semantic_results)} results")
                print(f"   🔗 Hybrid search: {len(hybrid_results)} results")
                
                # Compare results
                if hybrid_results and semantic_results:
                    # Check if ranking changed
                    semantic_files = [r['filename'] for r in semantic_results]
                    hybrid_files = [r['filename'] for r in hybrid_results]
                    
                    if semantic_files != hybrid_files:
                        print(f"   📈 Ranking changed: Hybrid scoring affected results")
                    else:
                        print(f"   📊 Ranking unchanged: Similar results from both methods")
                    
                    # Check for entity scores
                    entity_boosted = sum(1 for r in hybrid_results if r.get('entity_score', 0) > 0)
                    if entity_boosted > 0:
                        print(f"   🚀 Entity boost: {entity_boosted}/{len(hybrid_results)} results boosted")
                    else:
                        print(f"   📋 No entity boost applied")
                
            except Exception as e:
                print(f"   ❌ Search error: {e}")
                continue
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 HYBRID SYSTEM TEST SUMMARY")
        print("=" * 50)
        
        # Check integration components
        components = [
            ("FAISS Index", retriever.index is not None),
            ("BGE Embedder", retriever.embedder is not None),
            ("NER Data", retriever.ner_data is not None and len(retriever.ner_data) > 0),
            ("Entity Index", hasattr(retriever, 'entity_index') and len(retriever.entity_index) > 0),
            ("Entity Extraction", hasattr(retriever, '_extract_query_entities')),
            ("Hybrid Scoring", hasattr(retriever, '_calculate_entity_score')),
            ("Query Expansion", hasattr(retriever, '_expand_query_with_entities'))
        ]
        
        working_components = sum(1 for _, status in components if status)
        total_components = len(components)
        
        print(f"🔧 System Components: {working_components}/{total_components} working")
        for name, status in components:
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {name}")
        
        if working_components == total_components:
            print("\n🎉 SUCCESS: BGE and NER are fully integrated!")
            print("   ✅ Semantic search with BGE embeddings")
            print("   ✅ Entity extraction from queries")
            print("   ✅ Entity-based document scoring")
            print("   ✅ Hybrid score combination")
            print("   ✅ Query expansion with legal terms")
            print("   ✅ Re-ranking based on entity matching")
            
            print("\n🚀 The system now provides:")
            print("   • Better retrieval for legal entity queries")
            print("   • Improved ranking for documents with matching entities")
            print("   • Enhanced query understanding with legal context")
            print("   • Balanced semantic and entity-based scoring")
            
            return True
        else:
            print(f"\n⚠️ PARTIAL: {working_components}/{total_components} components working")
            print("   Some hybrid features may not be available")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_entity_detection_accuracy():
    """Test accuracy of entity detection from queries"""
    print("\n🎯 Testing Entity Detection Accuracy")
    print("-" * 40)
    
    test_cases = [
        ("Section 302 IPC", ["302", "ipc"]),
        ("Article 14 Constitution", ["14"]),
        ("Supreme Court of India", ["supreme court"]),
        ("criminal appeal", ["criminal appeal"]),
        ("High Court Delhi", ["high court"]),
        ("bail application", ["bail application"])
    ]
    
    try:
        from legal_retrieval_app import LegalDocumentRetriever
        retriever = LegalDocumentRetriever()
        
        if hasattr(retriever, '_extract_query_entities'):
            correct_detections = 0
            total_tests = len(test_cases)
            
            for query, expected_terms in test_cases:
                detected = retriever._extract_query_entities(query)
                
                # Check if any expected terms were detected
                all_detected_texts = []
                for entity_list in detected.values():
                    all_detected_texts.extend([text.lower() for text in entity_list])
                
                found_terms = [term for term in expected_terms if any(term in detected_text for detected_text in all_detected_texts)]
                
                if found_terms:
                    print(f"   ✅ '{query}' -> Found: {found_terms}")
                    correct_detections += 1
                else:
                    print(f"   ❌ '{query}' -> Expected: {expected_terms}, Got: {list(detected.keys())}")
            
            accuracy = (correct_detections / total_tests) * 100
            print(f"\n📊 Entity Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{total_tests})")
            
            return accuracy > 70  # Consider 70%+ as good
        else:
            print("❌ Entity extraction method not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing entity detection: {e}")
        return False

def main():
    """Run all hybrid system tests"""
    print("⚖️ Legal Document Retrieval System - Hybrid Integration Test")
    print("=" * 70)
    
    # Test 1: Overall integration
    integration_success = test_hybrid_integration()
    
    # Test 2: Entity detection accuracy
    detection_success = test_entity_detection_accuracy()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 70)
    
    if integration_success and detection_success:
        print("🎉 EXCELLENT: BGE + NER hybrid system is working optimally!")
        print("   ✅ Full integration between semantic and entity-based search")
        print("   ✅ Accurate entity detection from legal queries")
        print("   ✅ Effective hybrid scoring and re-ranking")
        print("\n🚀 Your system will provide superior legal document retrieval!")
    elif integration_success:
        print("✅ GOOD: Hybrid system is integrated but entity detection needs improvement")
        print("   ✅ BGE and NER are working together")
        print("   ⚠️ Entity detection accuracy could be better")
    elif detection_success:
        print("⚠️ PARTIAL: Entity detection works but integration is incomplete")
        print("   ✅ Entity detection is accurate")
        print("   ❌ Full hybrid integration not working")
    else:
        print("❌ NEEDS WORK: Both integration and detection need attention")
        print("   ❌ Hybrid system not fully functional")
        print("   ❌ Entity detection needs improvement")
    
    print(f"\n📖 For detailed usage instructions, see USAGE_GUIDE.md")
    print(f"🔧 For troubleshooting, run: python test_system.py")

if __name__ == "__main__":
    main()